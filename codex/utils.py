import torch

#  STABLE ATTENTION MASKING
def create_attention_mask(input_ids, pad_token_id, device):
    batch_size, seq_len = input_ids.shape

    # Use -1e9 instead of -inf for numerical stability
    mask_value = -1e9

    # Causal (look-ahead) mask
    mask = torch.full((seq_len, seq_len), mask_value, device=device)
    mask = torch.triu(mask, diagonal=1)
    combined_mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

    # Padding mask
    # (batch, 1, 1, seq_len)
    padding_mask = (input_ids == pad_token_id).float().unsqueeze(1).unsqueeze(2) * mask_value

    # Combine masks
    combined_mask = combined_mask + padding_mask
    return combined_mask

def generate_causal_mask(seq_len, device):
    mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
