import tensorflow as tf
from tensorflow.keras import layers

# --- Utility Layers and Functions ---
class LayerNorm(layers.Layer):
  def __init__(self, epsilon=1e-5, **kwargs):
    super().__init__(**kwargs)
    self.layernorm = layers.LayerNormalization(epsilon=epsilon)
  def call(self, inputs):
    return self.layernorm(inputs)

class GatingLayer(layers.Layer):
  def __init__(self, c_in, **kwargs):
    super().__init__(**kwargs)
    self.gate_linear = layers.Dense(c_in, activation=tf.nn.sigmoid)
  def call(self, inputs, gate_inputs=None):
    if gate_inputs is None:
      gate = self.gate_linear(inputs)
    else:
      gate = self.gate_linear(gate_inputs)
    return inputs * gate

class RotationTranslationUtilities:
  @staticmethod
  def create_rotation_matrix_from_euler(angles, dtype=tf.float32):
    alpha, beta, gamma = angles[0], angles[1], angles[2]
    c1 = tf.cos(alpha)
    s1 = tf.sin(alpha)
    c2 = tf.cos(beta)
    s2 = tf.sin(beta)
    c3 = tf.cos(gamma)
    s3 = tf.sin(gamma)
    R_x = tf.stack([
        tf.stack([tf.constant(1.0, dtype=dtype), tf.constant(0.0, dtype=dtype), tf.constant(0.0, dtype=dtype)]),
        tf.stack([tf.constant(0.0, dtype=dtype), c1, -s1]),
        tf.stack([tf.constant(0.0, dtype=dtype), s1, c1])
    ])
    R_y = tf.stack([
        tf.stack([c2, tf.constant(0.0, dtype=dtype), s2]),
        tf.stack([tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype), tf.constant(0.0, dtype=dtype)]),
        tf.stack([-s2, tf.constant(0.0, dtype=dtype), c2])
    ])
    R_z = tf.stack([
            tf.stack([c3, -s3, tf.constant(0.0, dtype=dtype)]),
            tf.stack([s3, c3, tf.constant(0.0, dtype=dtype)]),
            tf.stack([tf.constant(0.0, dtype=dtype), tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)])
    ])
    return tf.matmul(R_z, tf.matmul(R_y, R_x))

  @staticmethod
  def rotate_vector(rotation_matrix, vector):
    # rotation_matrix: (..., 3, 3), vector: (..., 3)
    # Ensure vector is expanded to (..., 3, 1) for batched matmul
    vector_expanded = tf.expand_dims(vector, axis=-1)
    result = tf.linalg.matmul(rotation_matrix, vector_expanded)
    return tf.squeeze(result, axis=-1) # Squeeze back to (..., 3)

  @staticmethod
  def translate_point(point, translation_vector):
    return point + translation_vector

  @staticmethod
  def translate_global_to_local(global_point, local_frame_rotation_matrix, local_frame_translation_vector):
    translated_point = global_point - local_frame_translation_vector
    # Transpose only the last two dimensions for inverse rotation, handling potential batch dimensions
    return RotationTranslationUtilities.rotate_vector(tf.transpose(local_frame_rotation_matrix, perm=[*range(tf.rank(local_frame_rotation_matrix) - 2), 2, 1]), translated_point)

  @staticmethod
  def transform_local_to_global(local_point, local_frame_rotation_matrix, local_frame_translation_vector):
    rotated_point = RotationTranslationUtilities.rotate_vector(local_frame_rotation_matrix, local_point)
    return rotated_point + local_frame_translation_vector


# --- Evoformer Sub-components ---
class MSAAttention(layers.Layer):
  def __init__(self, c_msa, c_hidden, num_heads, orientation='row', **kwargs):
    super().__init__(**kwargs)
    self.c_msa = c_msa
    self.c_hidden = c_hidden
    self.num_heads = num_heads
    self.orientation = orientation

    self.q_proj = layers.Dense(num_heads * c_hidden, use_bias=False)
    self.k_proj = layers.Dense(num_heads * c_hidden, use_bias=False)
    self.v_proj = layers.Dense(num_heads * c_hidden, use_bias=False)
    self.o_proj = layers.Dense(c_msa)
    self.gating_layer = GatingLayer(num_heads * c_hidden)
    self.layer_norm = LayerNorm()

    # For pairwise bias in row attention, if it exists
    if orientation == 'row':
      self.pairwise_bias_proj = layers.Dense(num_heads, use_bias=False)

  def call(self, msa_input, pairwise_bias=None, msa_mask=None):
    residual = msa_input
    norm_msa_input = self.layer_norm(msa_input)

    if self.orientation == 'row':
      B, N_seq, L, _ = tf.shape(norm_msa_input)

      q = self.q_proj(norm_msa_input)
      k = self.k_proj(norm_msa_input)
      v = self.v_proj(norm_msa_input)

      q = tf.reshape(q, [B, N_seq, L, self.num_heads, self.c_hidden])
      k = tf.reshape(k, [B, N_seq, L, self.num_heads, self.c_hidden])
      v = tf.reshape(v, [B, N_seq, L, self.num_heads, self.c_hidden])

      q = tf.transpose(q, [0, 3, 1, 2, 4]) # (B, num_heads, N_seq, L, c_hidden)
      k = tf.transpose(k, [0, 3, 1, 2, 4]) # (B, num_heads, N_seq, L, c_hidden)
      v = tf.transpose(v, [0, 3, 1, 2, 4]) # (B, num_heads, N_seq, L, c_hidden)

      # Attention: (B, H, N_seq, L_i, D) x (B, H, N_seq, L_k, D) -> (B, H, N_seq, L_i, L_k)
      attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.c_hidden, tf.float32))

      if pairwise_bias is not None:
        # pairwise_bias: (B, L, L, num_heads)
        # project to (B, L, L, num_heads)
        pairwise_bias_proj_out = self.pairwise_bias_proj(pairwise_bias) # (B, L, L, num_heads)
        pairwise_bias_proj_out = tf.transpose(pairwise_bias_proj_out, [0, 3, 1, 2]) # (B, num_heads, L, L)
        # Expand dims to match (B, num_heads, N_seq, L, L) for broadcasting (over N_seq)
        attention_scores += tf.expand_dims(pairwise_bias_proj_out, axis=2)

      if msa_mask is not None:
        # msa_mask shape: (B, N_seq, L). Need to expand to (B, 1, N_seq, 1, L) for broadcasting
        mask = tf.expand_dims(tf.expand_dims(msa_mask, axis=1), axis=3) # (B, 1, N_seq, 1, L)
        attention_scores = attention_scores + (1.0 - mask) * -1e9 # Fill masked positions with large negative value

      attention_weights = tf.nn.softmax(attention_scores, axis=-1)
      output = tf.matmul(attention_weights, v)
      output = tf.transpose(output, [0, 2, 3, 1, 4]) # (B, N_seq, L, num_heads, c_hidden)
      output = tf.reshape(output, [B, N_seq, L, self.num_heads * self.c_hidden])

    elif self.orientation == 'col':
      # Transpose to (B, L, N_seq, C_msa) for column attention
      msa_input_transposed = tf.transpose(norm_msa_input, [0, 2, 1, 3])
      B, L_seq, N_seq_msa, _ = tf.shape(msa_input_transposed)

      q = self.q_proj(msa_input_transposed)
      k = self.k_proj(msa_input_transposed)
      v = self.v_proj(msa_input_transposed)

      q = tf.reshape(q, [B, L_seq, N_seq_msa, self.num_heads, self.c_hidden])
      k = tf.reshape(k, [B, L_seq, N_seq_msa, self.num_heads, self.c_hidden])
      v = tf.reshape(v, [B, L_seq, N_seq_msa, self.num_heads, self.c_hidden])

      q = tf.transpose(q, [0, 3, 1, 2, 4]) # (B, num_heads, L, N_seq, c_hidden)
      k = tf.transpose(k, [0, 3, 1, 2, 4]) # (B, num_heads, L, N_seq, c_hidden)
      v = tf.transpose(v, [0, 3, 1, 2, 4]) # (B, num_heads, L, N_seq, c_hidden)

      attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.c_hidden, tf.float32))

      if msa_mask is not None:
        # msa_mask shape: (B, N_seq, L). Need col_mask (B, L, N_seq).
        col_mask = tf.transpose(msa_mask, [0, 2, 1]) # (B, L, N_seq)
        mask = tf.expand_dims(tf.expand_dims(col_mask, axis=1), axis=3) # (B, 1, L, 1, N_seq)
        attention_scores = attention_scores + (1.0 - mask) * -1e9

      attention_weights = tf.nn.softmax(attention_scores, axis=-1)
      output = tf.matmul(attention_weights, v)
      output = tf.transpose(output, [0, 2, 3, 1, 4]) # (B, L, N_seq, num_heads, c_hidden)
      output = tf.reshape(output, [B, L_seq, N_seq_msa, self.num_heads * self.c_hidden])
      output = tf.transpose(output, [0, 2, 1, 3]) # Transpose back to (B, N_seq, L, C_msa)

    else:
       raise ValueError(f"Invalid orientation: {self.orientation}. Must be 'row' or 'col'.")

    output = self.gating_layer(output)
    output = self.o_proj(output)
    return residual + output
  

class TransitionLayer(layers.Layer):
  def __init__(self, c_in, c_hidden, activation=tf.nn.gelu, **kwargs):
    super().__init__(**kwargs)
    self.linear_1 = layers.Dense(c_hidden, activation=activation)
    self.linear_2 = layers.Dense(c_in)
  def call(self, inputs):
    return self.linear_2(self.linear_1(inputs))

class OuterProductMean(layers.Layer):
  def __init__(self, c_msa, c_out, c_pair, **kwargs):
    super().__init__(**kwargs)
    self.c_msa = c_msa
    self.c_out = c_out
    self.c_pair = c_pair
    self.projection_a = layers.Dense(self.c_out)
    self.projection_b = layers.Dense(self.c_out)
    self.final_projection = layers.Dense(self.c_pair)

  def call(self, msa_representation, msa_mask):
    N_seq, L, _ = tf.shape(msa_representation)
    project_a_output = self.projection_a(msa_representation)
    project_b_output = self.projection_b(msa_representation)

    mask_expanded = tf.expand_dims(msa_mask, axis=-1)
    project_a_output *= mask_expanded
    project_b_output *= mask_expanded

    outer_products = tf.einsum('s i a, s j b -> s i j a b', project_a_output, project_b_output)
    summed_outer_products = tf.reduce_sum(outer_products, axis=0)

    valid_pair_count = tf.einsum('s i, s j -> i j', msa_mask, msa_mask)
    valid_pair_count = tf.maximum(valid_pair_count, 1e-6)
    valid_pair_count = tf.cast(tf.expand_dims(tf.expand_dims(valid_pair_count, axis=-1), axis=-1), tf.float32)

    averaged_outer_products = summed_outer_products / valid_pair_count

    reshaped_outer_products = tf.reshape(averaged_outer_products, [L, L, self.c_out * self.c_out])
    return self.final_projection(reshaped_outer_products)

class TriangularSelfAttention(layers.Layer):
  def __init__(self, c_pair, c_hidden, num_heads, orientation="outgoing", **kwargs):
    super().__init__(**kwargs)
    self.c_pair = c_pair
    self.c_hidden = c_hidden
    self.num_heads = num_heads
    if orientation not in ['outgoing', 'incoming']:
        raise ValueError(f"Invalid orientation: {orientation}. Must be 'outgoing' or 'incoming'.")
    self.orientation = orientation

    self.q_proj = layers.Dense(num_heads * c_hidden, use_bias=False)
    self.k_proj = layers.Dense(num_heads * c_hidden, use_bias=False)
    self.v_proj = layers.Dense(num_heads * c_hidden, use_bias=False)
    self.bias_proj = layers.Dense(num_heads, use_bias=False)
    self.o_proj = layers.Dense(c_pair)
    self.gating_layer = GatingLayer(num_heads * c_hidden)
    self.layer_norm = LayerNorm()

  def call(self, pair_representation, pair_mask=None):
    residual = pair_representation
    norm_pair_representation = self.layer_norm(pair_representation)

    L = tf.shape(pair_representation)[0]

    q = self.q_proj(norm_pair_representation)
    k = self.k_proj(norm_pair_representation)
    v = self.v_proj(norm_pair_representation)

    q = tf.reshape(q, [L, L, self.num_heads, self.c_hidden])
    k = tf.reshape(k, [L, L, self.num_heads, self.c_hidden])
    v = tf.reshape(v, [L, L, self.num_heads, self.c_hidden])

    q_t = tf.transpose(q, [2, 0, 1, 3])
    k_t = tf.transpose(k, [2, 0, 1, 3])
    v_t = tf.transpose(v, [2, 0, 1, 3])

    pairwise_bias_matrix = self.bias_proj(norm_pair_representation)

    if self.orientation == 'outgoing':
      attention_scores = tf.einsum('hijd,hikd->hijk', q_t, k_t) / tf.math.sqrt(tf.cast(self.c_hidden, tf.float32))

      bias_term_outgoing = tf.transpose(pairwise_bias_matrix, [1, 0, 2])
      bias_term_outgoing = tf.transpose(bias_term_outgoing, [2, 0, 1])
      attention_scores += tf.expand_dims(bias_term_outgoing, axis=1)

      if pair_mask is not None:
        key_mask_for_attn = tf.expand_dims(tf.cast(pair_mask, tf.float32), axis=0)
        key_mask_for_attn = tf.expand_dims(key_mask_for_attn, axis=2)
        attention_scores = attention_scores + (1.0 - key_mask_for_attn) * -1e9

      attention_weights = tf.nn.softmax(attention_scores, axis=-1)
      output = tf.matmul(attention_weights, v_t)
      output = tf.transpose(output, [1, 2, 0, 3])
      output = tf.reshape(output, [L, L, self.num_heads * self.c_hidden])

    elif self.orientation == 'incoming':
      k_t_for_incoming = tf.transpose(k_t, [0, 2, 1, 3])
      v_t_for_incoming = tf.transpose(v_t, [0, 2, 1, 3])

      attention_scores = tf.einsum('hijd,hkjd->hijk', q_t, k_t_for_incoming) / tf.math.sqrt(tf.cast(self.c_hidden, tf.float32))

      bias_term_incoming = tf.transpose(pairwise_bias_matrix, [2, 0, 1])
      attention_scores += tf.expand_dims(bias_term_incoming, axis=2)

      if pair_mask is not None:
        key_mask_for_attn = tf.transpose(tf.cast(pair_mask, tf.float32), [1, 0])
        key_mask_for_attn = tf.expand_dims(key_mask_for_attn, axis=0)
        key_mask_for_attn = tf.expand_dims(key_mask_for_attn, axis=1)
        attention_scores = attention_scores + (1.0 - key_mask_for_attn) * -1e9

      attention_weights = tf.nn.softmax(attention_scores, axis=-1)

      output = tf.einsum('hijk,hkjd->hijd', attention_weights, v_t_for_incoming)
      output = tf.transpose(output, [1, 2, 0, 3])
      output = tf.reshape(output, [L, L, self.num_heads * self.c_hidden])

    output = self.gating_layer(output)
    output = self.o_proj(output)
    return residual + output

class EvoformerBlock(layers.Layer):
  def __init__(self, c_msa, c_pair, c_hidden_msa_att, num_heads_msa, c_hidden_opm, c_hidden_tri_att, num_heads_tri, **kwargs):
    super().__init__(**kwargs)
    self.c_msa = c_msa
    self.c_pair = c_pair
    self.msa_row_attention = MSAAttention(c_msa=c_msa, c_hidden=c_hidden_msa_att, num_heads=num_heads_msa, orientation='row')
    self.msa_col_attention = MSAAttention(c_msa=c_msa, c_hidden=c_hidden_msa_att, num_heads=num_heads_msa, orientation='col')
    self.msa_transition = TransitionLayer(c_in=c_msa, c_hidden=c_msa*4)
    self.msa_transition_ln = LayerNorm()
    self.outer_product_mean = OuterProductMean(c_msa=c_msa, c_out=c_hidden_opm, c_pair=c_pair)
    self.outer_product_mean_ln = LayerNorm()
    self.triangular_self_attention_outgoing = TriangularSelfAttention(c_pair=c_pair, c_hidden=c_hidden_tri_att, num_heads=num_heads_tri, orientation='outgoing')
    self.triangular_self_attention_incoming = TriangularSelfAttention(c_pair=c_pair, c_hidden=c_hidden_tri_att, num_heads=num_heads_tri, orientation='incoming')
    self.pair_transition = TransitionLayer(c_in=c_pair, c_hidden=c_pair*4)
    self.pair_transition_ln = LayerNorm()

  def call(self, msa_representation, pair_representation, msa_mask, pair_mask=None):
    msa_representation = self.msa_row_attention(msa_representation, pairwise_bias=pair_representation, msa_mask=msa_mask)
    msa_representation = self.msa_col_attention(msa_representation, msa_mask=msa_mask)
    msa_representation = msa_representation + self.msa_transition(self.msa_transition_ln(msa_representation))

    pair_update_from_msa = self.outer_product_mean(self.outer_product_mean_ln(msa_representation), msa_mask)
    pair_representation = pair_representation + pair_update_from_msa

    pair_representation = self.triangular_self_attention_outgoing(pair_representation, pair_mask=pair_mask)
    pair_representation = self.triangular_self_attention_incoming(pair_representation, pair_mask=pair_mask)
    pair_representation = pair_representation + self.pair_transition(self.pair_transition_ln(pair_representation))

    return msa_representation, pair_representation


# --- Invariant Point Attention (IPA) Layer ---
class InvariantPointAttention(layers.Layer):
  def __init__(
      self,
      c_in,
      c_hidden_scalar,
      c_hidden_point,
      num_heads,
      num_points,
      **kwargs
  ):
    super().__init__(**kwargs)
    self.c_in=c_in
    self.c_hidden_scalar=c_hidden_scalar
    self.c_hidden_point=c_hidden_point
    self.num_heads=num_heads
    self.num_points=num_points

    self.layer_norm=LayerNorm()

    self.q_scalar_proj=layers.Dense(num_heads*c_hidden_scalar,use_bias=False)
    self.k_scalar_proj=layers.Dense(num_heads*c_hidden_scalar,use_bias=False)
    self.v_scalar_proj=layers.Dense(num_heads*c_hidden_scalar,use_bias=False)

    self.q_point_proj=layers.Dense(num_heads*num_points*3,use_bias=False)
    self.k_point_proj=layers.Dense(num_heads*num_points*3,use_bias=False)
    self.v_point_proj=layers.Dense(num_heads*num_points*3,use_bias=False)

    self.attention_bias_proj=layers.Dense(num_heads,use_bias=False)

    output_gating_dim=num_heads*c_hidden_scalar + num_heads*num_points*3
    self.gating_layer=GatingLayer(output_gating_dim)
    self.out_proj=layers.Dense(c_in)

  def call(self, query_features, pairwise_features, frames, attention_mask=None):
    L = tf.shape(query_features)[-2] # Batch, L, C_in

    rotations = tf.reshape(frames[..., :9], [tf.shape(frames)[0], L, 3, 3])
    translations = frames[..., 9:]

    norm_query_features = self.layer_norm(query_features)

    q_scalar = self.q_scalar_proj(norm_query_features)
    k_scalar = self.k_scalar_proj(norm_query_features)
    v_scalar = self.v_scalar_proj(norm_query_features)

    q_scalar = tf.reshape(q_scalar, [tf.shape(q_scalar)[0], L, self.num_heads, self.c_hidden_scalar])
    k_scalar = tf.reshape(k_scalar, [tf.shape(k_scalar)[0], L, self.num_heads, self.c_hidden_scalar])
    v_scalar = tf.reshape(v_scalar, [tf.shape(v_scalar)[0], L, self.num_heads, self.c_hidden_scalar])

    q_point_flat = self.q_point_proj(norm_query_features)
    k_point_flat = self.k_point_proj(norm_query_features)
    v_point_flat = self.v_point_proj(norm_query_features)

    q_point = tf.reshape(q_point_flat, [tf.shape(q_point_flat)[0], L, self.num_heads, self.num_points, 3])
    k_point = tf.reshape(k_point_flat, [tf.shape(k_point_flat)[0], L, self.num_heads, self.num_points, 3])
    v_point = tf.reshape(v_point_flat, [tf.shape(v_point_flat)[0], L, self.num_heads, self.num_points, 3])

    attention_bias = self.attention_bias_proj(pairwise_features) # (B, L, L, num_heads)

    scalar_attention_scores = tf.einsum('bihd,bjhd->bijh', q_scalar, k_scalar) / tf.math.sqrt(tf.cast(self.c_hidden_scalar, tf.float32))

    translations_expanded = tf.expand_dims(tf.expand_dims(translations, axis=2), axis=2) # (B, L, 1, 1, 3)

    k_point_expanded_for_sub = tf.expand_dims(k_point, axis=1) # (B, 1, L, H, P, 3)
    k_point_global_translated = k_point_expanded_for_sub - translations_expanded # (B, L, L, H, P, 3)

    rotations_transposed = tf.transpose(rotations, perm=[0, 1, 3, 2]) # (B, L, 3, 3)

    local_k_points = RotationTranslationUtilities.rotate_vector(
        tf.expand_dims(rotations_transposed, axis=2), # (B, L, 1, 3, 3)
        k_point_global_translated # (B, L, L, H, P, 3)
    )

    local_q_points = tf.expand_dims(q_point, axis=2) # (B, L, 1, H, P, 3)

    squared_diff = tf.square(local_q_points - local_k_points)
    sum_squared_diff_per_point = tf.reduce_sum(squared_diff, axis=-1) # (B, L, L, H, P)
    sum_across_points = tf.reduce_sum(sum_squared_diff_per_point, axis=-1) # (B, L, L, H)

    point_attention_scores = -0.5 * sum_across_points

    total_attention_scores = scalar_attention_scores + point_attention_scores + attention_bias

    if attention_mask is not None:
        mask_expanded = tf.expand_dims(tf.cast(attention_mask, tf.float32), axis=-1) # (B, L, L, 1)
        total_attention_scores = total_attention_scores + (1.0 - mask_expanded) * -1e9

    attention_weights = tf.nn.softmax(total_attention_scores, axis=2) # Attention over keys (axis=2 for j)

    weighted_scalar_output = tf.einsum('bijh,bjhd->bihd', attention_weights, v_scalar) # (B, L, H, D)

    rotations_expanded = tf.expand_dims(tf.expand_dims(rotations, axis=2), axis=2) # (B, L, 1, 1, 3, 3)
    translations_expanded_v = tf.expand_dims(tf.expand_dims(translations, axis=2), axis=2) # (B, L, 1, 1, 3)

    v_point_expanded_for_sub = tf.expand_dims(v_point, axis=1) # (B, 1, L, H, P, 3)
    v_point_global_translated = v_point_expanded_for_sub - translations_expanded_v # (B, L, L, H, P, 3)

    local_v_points = RotationTranslationUtilities.rotate_vector(
        tf.expand_dims(rotations_transposed, axis=2), # (B, L, 1, 3, 3)
        v_point_global_translated # (B, L, L, H, P, 3)
    ) # (B, L, L, H, P, 3)

    attention_weights_expanded = tf.expand_dims(tf.expand_dims(attention_weights, axis=-1), axis=-1) # (B, L, L, H, 1, 1)

    weighted_local_point_output = tf.reduce_sum(local_v_points * attention_weights_expanded,
                                                axis=2) # Sum over j (key dimension) -> (B, L, H, P, 3)

    global_weighted_point_output_rotated = RotationTranslationUtilities.rotate_vector(
        tf.expand_dims(rotations, axis=2), # (B, L, 1, 3, 3)
        weighted_local_point_output # (B, L, H, P, 3)
    ) # (B, L, H, P, 3)

    translations_expanded_for_point = tf.expand_dims(tf.expand_dims(translations, axis=2), axis=2) # (B, L, 1, 1, 3)
    global_weighted_point_output = global_weighted_point_output_rotated + translations_expanded_for_point

    combined_attention_output = tf.concat([
        tf.reshape(weighted_scalar_output, [tf.shape(weighted_scalar_output)[0], L, self.num_heads * self.c_hidden_scalar]),
        tf.reshape(global_weighted_point_output, [tf.shape(global_weighted_point_output)[0], L, self.num_heads * self.num_points * 3])
    ], axis=-1)

    gated_output = self.gating_layer(combined_attention_output)
    output_features = self.out_proj(gated_output) + query_features # (B, L, C_in)

    delta_translation = tf.reduce_sum(
        tf.einsum('bijh,bihp->bihp', attention_weights, q_point), # (B, L, L, H) x (B, L, H, P, 3) -> (B, L, H, P, 3)
        axis=2 # Sum over L from q_point
    )
    delta_translation = tf.reduce_sum(delta_translation, axis=2) # Sum over heads -> (B, L, 3)

    delta_rotation = tf.zeros((tf.shape(query_features)[0], L, 3, 3), dtype=tf.float32)

    return output_features, delta_translation, delta_rotation


# --- Evoformer and Structure Module (Dummy) ---
class DummyModelPass(layers.Layer):
    def __init__(self, c_msa, c_pair, c_in_ipa, c_hidden_scalar_ipa, c_hidden_point_ipa, num_heads_ipa, num_points_ipa, **kwargs):
        super().__init__(**kwargs)
        self.c_msa = c_msa
        self.c_pair = c_pair

        self.evoformer_block = EvoformerBlock(
            c_msa=c_msa, c_pair=c_pair,
            c_hidden_msa_att=64, num_heads_msa=8,
            c_hidden_opm=64,
            c_hidden_tri_att=64, num_heads_tri=8
        )

        self.initial_frames_proj = layers.Dense(12)
        self.initial_coords_proj = layers.Dense(3)
        self.ipa_layer = InvariantPointAttention(
            c_in=c_in_ipa,
            c_hidden_scalar=c_hidden_scalar_ipa,
            c_hidden_point=c_hidden_point_ipa,
            num_heads=num_heads_ipa,
            num_points=num_points_ipa
        )
        self.ipa_res_proj = layers.Dense(c_in_ipa)

    def call(self, msa_input, pair_input, prev_frames, prev_coords, template_features=None, msa_mask=None, pair_mask=None):
        msa_input_shape = tf.shape(msa_input)
        B = msa_input_shape[0]
        N_seq = msa_input_shape[1]
        L = msa_input_shape[2]

        prev_coords_shape = tf.shape(prev_coords)
        N_atom = tf.cond(tf.equal(tf.rank(prev_coords), 4),
                         lambda: prev_coords_shape[-2],
                         lambda: tf.constant(5, dtype=tf.int32))

        msa_out, pair_out = self.evoformer_block(msa_input, pair_input, msa_mask=msa_mask, pair_mask=pair_mask)

        query_residue_features = msa_out[:, 0, :, :] # (B, L, C_msa)
        query_residue_features_ipa_in = self.ipa_res_proj(query_residue_features) # (B, L, c_in_ipa)

        initial_frames = tf.cond(tf.reduce_all(tf.equal(prev_frames, 0.0)),
                                 lambda: self.initial_frames_proj(query_residue_features), # (B, L, 12)
                                 lambda: prev_frames)

        initial_coords = tf.cond(tf.reduce_all(tf.equal(prev_coords, 0.0)),
                                 lambda: tf.expand_dims(self.initial_coords_proj(query_residue_features), axis=2) + tf.zeros((B, L, N_atom - 1, 3), dtype=tf.float32),
                                 lambda: prev_coords)

        updated_residue_features, delta_translation, delta_rotation = self.ipa_layer(
            query_features=query_residue_features_ipa_in,
            pairwise_features=pair_out,
            frames=initial_frames,
            attention_mask=pair_mask
        )

        updated_translations = initial_frames[..., 9:] + delta_translation
        initial_rotations = tf.reshape(initial_frames[..., :9], [B, L, 3, 3])
        updated_rotations_flat = tf.reshape(initial_rotations, [B, L, 9])
        updated_frames = tf.concat([updated_rotations_flat, updated_translations], axis=-1)

        predicted_coords = tf.expand_dims(updated_translations, axis=-2) + tf.random.normal((B, L, N_atom, 3), stddev=0.1)

        return msa_out, pair_out, updated_frames, predicted_coords


# --- Recycling Mechanism ---
class RecyclingBlock(layers.Layer):
    def __init__(self, model_components, num_recycling_steps, c_msa, c_pair, c_in_ipa, **kwargs):
        super().__init__(**kwargs)
        self.model_components = model_components
        self.num_recycling_steps = num_recycling_steps
        self.c_msa = c_msa
        self.c_pair = c_pair
        self.c_in_ipa = c_in_ipa

        self.zero_recycled_msa = self.add_weight(
            name="zero_recycled_msa",
            shape=(1, c_msa), # (1, C_msa) for per-residue features of the recycled query
            initializer="zeros",
            trainable=True
        )
        self.zero_recycled_pair = self.add_weight(
            name="zero_recycled_pair",
            shape=(1, c_pair), # (1, C_pair) for per-pair features (will be tiled to L,L,C_pair)
            initializer="zeros",
            trainable=True
        )
        # AlphaFold also applies a dense layer to the raw MSA input before concatenation
        self.msa_input_proj = layers.Dense(c_msa)

    def call(self, msa_input_features, pair_input_features, template_features, msa_mask, pair_mask, atom_mask):
        B = tf.shape(msa_input_features)[0] # Batch size
        L = tf.shape(msa_input_features)[2] # Sequence length
        N_atom = tf.shape(atom_mask)[-1] # Number of atoms per residue

        # Initialize recycled components, tiled to batch_size and sequence length
        # prev_msa_output for the recycled query sequence (B, L, C_msa)
        prev_msa_output = tf.tile(tf.expand_dims(self.zero_recycled_msa, axis=0), [B, L, 1])
        # prev_pair_output (B, L, L, C_pair)
        prev_pair_output = tf.tile(tf.expand_dims(tf.expand_dims(self.zero_recycled_pair, axis=0), axis=0), [B, L, L, 1])

        prev_frames = tf.zeros((B, L, 12), dtype=tf.float32)
        prev_coords = tf.zeros((B, L, N_atom, 3), dtype=tf.float32)

        all_msa_outputs = []
        all_pair_outputs = []
        all_predicted_frames = []
        all_predicted_coords = []

        # Project raw MSA input features once per protein before recycling loop
        projected_msa_input = self.msa_input_proj(msa_input_features)

        for i in tf.range(self.num_recycling_steps):
            # Augment MSA input for Evoformer: Recycled query MSA is prepended to actual MSA input features
            current_msa_input_for_evoformer = tf.concat([tf.expand_dims(prev_msa_output, axis=1), projected_msa_input], axis=1) # (B, 1+N_seq, L, C_msa)
            # Create mask for the recycled MSA row (it's always valid)
            mask_for_recycled_msa_row = tf.ones((B, 1, L), dtype=tf.float32)
            current_msa_mask_for_evoformer = tf.concat([mask_for_recycled_msa_row, msa_mask], axis=1) # (B, 1+N_seq, L)

            # Augment Pair input: Add previous pairwise representation to current pairwise input
            current_pair_input_for_evoformer = pair_input_features + prev_pair_output # (B, L, L, C_pair)

            # Forward pass through the core model components
            msa_out, pair_out, predicted_frames, predicted_coords = self.model_components(
                msa_input=current_msa_input_for_evoformer,
                pair_input=current_pair_input_for_evoformer,
                prev_frames=prev_frames,
                prev_coords=prev_coords,
                template_features=template_features,
                msa_mask=current_msa_mask_for_evoformer,
                pair_mask=pair_mask
            )

            # Store outputs of current recycling step
            all_msa_outputs.append(msa_out)
            all_pair_outputs.append(pair_out)
            all_predicted_frames.append(predicted_frames)
            all_predicted_coords.append(predicted_coords)

            # Update recycled features for the next iteration
            # The first sequence of Evoformer output is the refined recycled query MSA
            prev_msa_output = msa_out[:, 0, :, :] # (B, L, C_msa)
            prev_pair_output = pair_out # (B, L, L, C_pair)
            prev_frames = predicted_frames # (B, L, 12)
            prev_coords = predicted_coords # (B, L, N_atom, 3)

        return (
            all_msa_outputs,
            all_pair_outputs,
            all_predicted_frames,
            all_predicted_coords
        )

