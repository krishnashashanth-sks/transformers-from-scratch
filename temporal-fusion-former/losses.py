import torch.nn as nn
import torch

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions, targets):
        # predictions: (batch_size, decoder_length, num_quantiles) -> e.g., (32, 10, 3)
        # targets: (batch_size, decoder_length) -> e.g., (32, 10)

        # Assert correct dimensionality
        assert predictions.dim() == 3, f"Predictions must be 3D (B, S, Q), but got {predictions.dim()}D shape {predictions.shape}"
        assert targets.dim() == 2, f"Targets must be 2D (B, S), but got {targets.dim()}D shape {targets.shape}"

        # Assert shapes match
        batch_size, decoder_length, num_quantiles = predictions.shape
        assert batch_size == targets.shape[0], f"Batch size mismatch: predictions {batch_size} vs targets {targets.shape[0]}"
        assert decoder_length == targets.shape[1], f"Sequence length mismatch: predictions {decoder_length} vs targets {targets.shape[1]}"
        assert num_quantiles == len(self.quantiles), f"Number of predictions quantiles ({num_quantiles}) must match defined quantiles ({len(self.quantiles)})"

        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract predictions for the current quantile
            pred_q = predictions[:, :, i]  # Shape: (batch_size, decoder_length)

            # Calculate errors. Both targets and pred_q are (B, S), so direct subtraction works.
            errors = targets - pred_q

            # Apply quantile loss formula element-wise
            q_loss = torch.max((q - 1) * errors, q * errors)
            losses.append(q_loss.unsqueeze(-1)) # Add back a dimension to stack later

        # Concatenate all quantile losses along a new dimension and take the mean
        # Resulting shape: (batch_size, decoder_length, num_quantiles)
        return torch.cat(losses, dim=-1).mean()

