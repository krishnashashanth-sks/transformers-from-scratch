import torch

def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Generates a causal mask (look-ahead mask) for self-attention.
    Ensures that position i can only attend to positions j <= i.
    """
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask