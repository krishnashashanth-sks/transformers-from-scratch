import torch.nn.functional as F
import torch

def hubert_loss(prediction_logits,discrete_targets,mask_indices):
  """
  Calculates the HuBERT pre-training loss (Cross-Entropy) on masked positions.

  Args:
      prediction_logits (torch.Tensor): Logits from the prediction head,
                                        shape (batch_size, time_steps, num_clusters).
      discrete_targets (np.ndarray): True discrete targets (cluster IDs) for all frames,
                                  shape (total_frames,).
      mask_indices (torch.BoolTensor): Boolean tensor indicating masked positions,
                                    shape (batch_size, time_steps).

  Returns:
      torch.Tensor: The calculated Cross-Entropy Loss.
  """
  batch_size,time_steps,num_cluster=prediction_logits.shape
  discrete_targets_tensor=torch.from_numpy(discrete_targets).long().to(prediction_logits.device)

  # Reshape discrete_targets_tensor to (batch_size, time_steps).
  # This assumes discrete_targets is a flat array where frames from each batch item are concatenated
  # and that all batch items have the same number of time_steps.
  true_targets_reshaped = discrete_targets_tensor.view(batch_size, time_steps)

  masked_logits=prediction_logits[mask_indices]
  masked_targets=true_targets_reshaped[mask_indices]
  # Use ignore_index=-1 to skip padded target values during loss calculation
  return F.cross_entropy(masked_logits,masked_targets, ignore_index=-1)