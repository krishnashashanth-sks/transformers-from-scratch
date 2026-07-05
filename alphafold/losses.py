import tensorflow as tf
from layers import RotationTranslationUtilities

# --- Loss Functions ---
def masked_msa_loss(true_labels, predicted_logits):
    predicted_len_L = tf.shape(predicted_logits)[1]

    # Flatten true_labels to (Batch * Num_masked_positions) and predicted_logits to (Batch * L, Vocab_size)
    # This assumes true_labels for masked positions are aligned with a portion of predicted_logits
    # For this dummy, we assume true_labels corresponds to the query sequence's length L.
    true_labels_flat = tf.reshape(true_labels, [-1]) # (B * L)
    predicted_logits_flat = tf.reshape(predicted_logits, [-1, tf.shape(predicted_logits)[-1]]) # (B * L, Vocab_size)

    # For the dummy true_masked_msa_labels, it's (Batch, num_masked_positions)
    # We need to ensure that the predicted_logits align with the target masked positions.
    # For simplicity, let's just use a slice of the predicted logits that corresponds to the true_labels shape.
    # In a real AF2, there would be a mask to select relevant logits for masked positions.
    num_masked_labels = tf.shape(true_labels_flat)[0]
    predicted_logits_for_masked = predicted_logits_flat[:num_masked_labels]

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    per_element_loss = loss_obj(true_labels_flat, predicted_logits_for_masked)
    return tf.reduce_mean(per_element_loss)

def distogram_loss(true_labels, predicted_logits):
    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    per_element_loss = loss_obj(true_labels, predicted_logits)
    return tf.reduce_mean(per_element_loss)

def experimental_constraint_loss(true_coords, predicted_coords, atom_mask):
    diff = tf.abs(true_coords - predicted_coords)
    masked_diff = diff * tf.expand_dims(atom_mask, axis=-1)
    return tf.reduce_sum(masked_diff) / (tf.reduce_sum(atom_mask) + 1e-6)

def plddt_loss(true_labels, predicted_logits):
    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    per_element_loss = loss_obj(true_labels, predicted_logits)
    return tf.reduce_mean(per_element_loss)

def fape_loss(
    pred_frames: tf.Tensor,
    pred_coords: tf.Tensor,
    true_frames: tf.Tensor,
    true_coords: tf.Tensor,
    residue_mask: tf.Tensor,
    atom_mask: tf.Tensor,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    ca_only: bool = False
) -> tf.Tensor:
    B = tf.shape(pred_frames)[0]
    L = tf.shape(pred_frames)[1]
    N_atom = tf.shape(pred_coords)[2]

    pred_rotations = tf.reshape(pred_frames[..., :9], [B, L, 3, 3])
    pred_translations = pred_frames[..., 9:]

    true_rotations = tf.reshape(true_frames[..., :9], [B, L, 3, 3])
    true_translations = true_frames[..., 9:]

    if ca_only:
        pred_coords_to_use = pred_coords[:, :, 0:1, :]
        true_coords_to_use = true_coords[:, :, 0:1, :]
        atom_mask_to_use = atom_mask[:, :, 0:1]
        N_atom_eff = 1
    else:
        pred_coords_to_use = pred_coords
        true_coords_to_use = true_coords
        atom_mask_to_use = atom_mask
        N_atom_eff = N_atom

    pred_translations_expanded = tf.expand_dims(pred_translations, axis=-2)
    translated_true_coords = true_coords_to_use - pred_translations_expanded
    pred_rotations_transposed = tf.transpose(pred_rotations, perm=[0, 1, 3, 2])

    local_true_coords_transformed = RotationTranslationUtilities.rotate_vector(
        tf.expand_dims(pred_rotations_transposed, axis=-2), # (B, L, 1, 3, 3)
        translated_true_coords # (B, L, N_atom_eff, 3)
    )

    distances = tf.norm(local_true_coords_transformed - pred_coords_to_use, axis=-1)
    clamped_distances = tf.minimum(distances, clamp_distance)
    fape_terms = clamped_distances / loss_unit_distance

    residue_mask_expanded = tf.expand_dims(tf.expand_dims(tf.cast(residue_mask, tf.float32), axis=-1), axis=-1) # (B, L, 1, 1)

    combined_mask = residue_mask_expanded * tf.expand_dims(tf.cast(atom_mask_to_use, tf.float32), axis=-1) # (B, L, N_atom_eff, 1)
    masked_fape_terms = fape_terms * tf.squeeze(combined_mask, axis=-1) # (B, L, N_atom_eff)

    sum_fape_terms = tf.reduce_sum(masked_fape_terms)
    total_unmasked_atoms = tf.reduce_sum(tf.squeeze(combined_mask, axis=-1))

    fape_loss_value = sum_fape_terms / (total_unmasked_atoms + 1e-6)

    return fape_loss_value

def composite_loss_fn(predictions, batch, num_recycling_steps):
    total_loss = tf.constant(0.0, dtype=tf.float32)
    individual_losses = {}

    w_fape = 0.5
    w_msa = 0.02
    w_distogram = 0.3
    w_constraints = 0.01
    w_plddt = 0.1

    fape_losses_per_recycle = []
    for i in range(num_recycling_steps):
        current_fape_loss = fape_loss(
            pred_frames=predictions['all_predicted_frames'][i],
            pred_coords=predictions['all_predicted_coords'][i],
            true_frames=batch['true_frames'],
            true_coords=batch['true_coords'],
            residue_mask=batch['residue_mask'],
            atom_mask=batch['atom_mask'],
            ca_only=False
        )
        total_loss += w_fape * current_fape_loss
        fape_losses_per_recycle.append(current_fape_loss)
    individual_losses['fape_loss'] = tf.reduce_mean(tf.stack(fape_losses_per_recycle))

    msa_pred_loss = masked_msa_loss(
        true_labels=batch['true_masked_msa_labels'],
        predicted_logits=predictions['masked_msa_logits']
    )
    total_loss += w_msa * msa_pred_loss
    individual_losses['masked_msa_loss'] = msa_pred_loss

    dist_loss = distogram_loss(
        true_labels=batch['true_distogram_labels'],
        predicted_logits=predictions['distogram_logits']
    )
    total_loss += w_distogram * dist_loss
    individual_losses['distogram_loss'] = dist_loss

    constraint_loss = experimental_constraint_loss(
        true_coords=batch['true_coords'],
        predicted_coords=predictions['final_predicted_coords'],
        atom_mask=batch['atom_mask']
    )
    total_loss += w_constraints * constraint_loss
    individual_losses['experimental_constraint_loss'] = constraint_loss

    plddt_pred_loss = plddt_loss(
        true_labels=batch['true_plddt_labels'],
        predicted_logits=predictions['plddt_logits']
    )
    total_loss += w_plddt * plddt_pred_loss
    individual_losses['plddt_loss'] = plddt_pred_loss

    return total_loss, individual_losses

