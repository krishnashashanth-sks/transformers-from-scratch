import torch
import torch.nn.functional as F

def dice_score(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    # inputs are logits, targets are ground truth masks (0 or 1)

    # Ensure inputs and targets have the same shape
    if inputs.shape != targets.shape:
        # Adjust targets if necessary (e.g., add channel dimension if missing)
        if targets.dim() == inputs.dim() - 1:
            targets = targets.unsqueeze(1) # Add channel dimension
        else:
            raise ValueError(f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}")

    # Apply sigmoid to convert logits to probabilities, then threshold to binary prediction
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float() # Binary prediction

    # Flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Calculate intersection and union
    intersection = (inputs * targets).sum()
    total_sum = inputs.sum() + targets.sum()

    # Calculate Dice score
    dice = (2. * intersection + smooth) / (total_sum + smooth)

    return dice

def iou_score(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    # inputs are logits, targets are ground truth masks (0 or 1)

    # Ensure inputs and targets have the same shape
    if inputs.shape != targets.shape:
        # Adjust targets if necessary (e.g., add channel dimension if missing)
        if targets.dim() == inputs.dim() - 1:
            targets = targets.unsqueeze(1) # Add channel dimension
        else:
            raise ValueError(f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}")

    # Apply sigmoid to convert logits to probabilities, then threshold to binary prediction
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float() # Binary prediction

    # Flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Calculate intersection and union
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)

    return iou
