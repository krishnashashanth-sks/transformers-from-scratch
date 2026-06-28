import torch
import torch.nn as nn
from layers import *

class Llama(nn.Module):
  def __init__(self,vocab_size,dim,num_layers,num_heads,hidden_size,max_seq_len,dropout_rate,seq_len_interpolation_factor=1.0):
    super().__init__()
    self.vocab_size=vocab_size
    self.dim=dim
    self.num_layers=num_layers
    self.num_heads=num_heads
    self.hidden_dim=hidden_size
    self.max_seq_len=max_seq_len
    self.dropout_rate=dropout_rate

    self.token_embeddings=nn.Embedding(vocab_size,dim)
    self.rope_layer=RotaryPositionalEmbedding(dim=dim//num_heads,seq_len_interpolation_factor=seq_len_interpolation_factor)

    self.layers=nn.ModuleList([
        LlamaDecoderBlock(
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_size,
            dropout_rate=dropout_rate,
            rope_layer=self.rope_layer
        )
        for _ in range(num_layers)
    ])

    self.norm=RMSNorm(dim)

  def forward(self,input_ids,padding_attention_mask=None):
    # input_ids: (batch_size, seq_len)
    # padding_attention_mask: (batch_size, seq_len) -> 1 for real tokens, 0 for padding

    x = self.token_embeddings(input_ids) # (batch_size, seq_len, dim)
    batch_size, seq_len, _ = x.shape

    # Create causal mask (lower triangular) (seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
    # Reshape to (1, 1, seq_len, seq_len) for broadcasting
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

    # Combine causal mask with padding attention mask
    # The padding_attention_mask is (batch_size, seq_len)
    # We need to expand it to (batch_size, 1, 1, seq_len) to broadcast correctly with causal_mask
    if padding_attention_mask is not None:
        # Expand padding mask to (batch_size, 1, 1, seq_len) to operate on the sequence dimension of causal_mask
        expanded_padding_mask = padding_attention_mask.unsqueeze(1).unsqueeze(2).bool()
        combined_mask = causal_mask & expanded_padding_mask
    else:
        combined_mask = causal_mask

    # The MultiHeadAttention expects mask of shape (batch_size, 1, seq_len, seq_len) or similar.
    # `combined_mask` has `dtype=bool`. `masked_fill` expects a boolean mask.
    # It's typically applied to `attn_scores` directly, where `False` means mask (set to -inf).
    # The `MultiHeadAttention`'s `masked_fill` uses `mask==0` (from float mask) or `~mask` (from bool mask).
    # Let's ensure the mask passed to MHA is consistent with its expectation.
    # Since the MHA's `masked_fill` checks `mask == 0` for a float mask,
    # passing `~combined_mask` (which is boolean) to indicate `True` for masked positions will work if we convert it to float.
    # Alternatively, the MHA can be updated to accept boolean mask directly.
    # For now, let's pass a boolean mask directly and assume MHA uses `~mask` or `mask == False` for masking.

    for layer in self.layers:
      # Pass the combined boolean mask to the decoder block
      x=layer(x, mask=combined_mask)

    return self.norm(x)

class LlamaForCausalLM(nn.Module):
  def __init__(self,llama_model,vocab_size):
    super().__init__()
    self.model=llama_model
    self.lm_head=nn.Linear(llama_model.dim,vocab_size,bias=False)
  def forward(self,input_ids,attention_mask=None):
    hidden_states=self.model(input_ids,padding_attention_mask=attention_mask)
    logits=self.lm_head(hidden_states)
    return logits