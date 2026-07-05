import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            if targets.dim() == inputs.dim() - 1:
                targets = targets.unsqueeze(1)
            else:
                raise ValueError(f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}")

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            if targets.dim() == inputs.dim() - 1:
                targets = targets.unsqueeze(1)
            else:
                raise ValueError(f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}")

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        loss = 1 - dice_score

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(nn.Module):
    def __init__(self, weight_focal: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.focal_loss = FocalLoss(reduction='mean')
        self.dice_loss = DiceLoss(reduction='mean')
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_l = self.focal_loss(inputs, targets)
        dice_l = self.dice_loss(inputs, targets)

        total_loss = self.weight_focal * focal_l + self.weight_dice * dice_l
        return total_loss
