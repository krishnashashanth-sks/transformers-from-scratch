import torch

def rotate_half(x):
  x1=x[...,:x.shape[-1]//2]
  x2=x[...,x.shape[-1]//2:]
  return torch.cat((-x2,x1),dim=-1)

def apply_rotary_pos_emb(q,k,cos,sin,position_ids):
  # cos and sin are (1, 1, seq_len, head_dim)
  # position_ids is (batch_size, seq_len)
  # We need to index the seq_len dimension of cos/sin using position_ids
  # and then unsqueeze to allow broadcasting with num_heads
  cos_indexed = cos.squeeze(0).squeeze(0)[position_ids].unsqueeze(1) # (batch_size, 1, seq_len, head_dim)
  sin_indexed = sin.squeeze(0).squeeze(0)[position_ids].unsqueeze(1) # (batch_size, 1, seq_len, head_dim)

  q_embed=(q*cos_indexed)+(rotate_half(q)*sin_indexed)
  k_embed=(k*cos_indexed)+(rotate_half(k)*sin_indexed)
  return q_embed,k_embed

