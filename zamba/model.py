import torch.nn as nn
from layers import MambaBlock,GlobalSharedAttention

class Zamba(nn.Module):
  def __init__(self, d_model, d_inner, d_state=16, num_mamba_blocks=12, attention_interval=6, vocab_size=None):
    super().__init__()
    self.d_model = d_model
    self.d_inner = d_inner
    self.d_state = d_state
    self.num_mamba_blocks = num_mamba_blocks
    self.attention_interval = attention_interval
    self.vocab_size = vocab_size

    # Add embedding layer
    if vocab_size is None:
        raise ValueError("vocab_size must be provided for Zamba model initialization.")
    self.embedding = nn.Embedding(vocab_size, d_model)

    self.layers = nn.ModuleList()
    for i in range(num_mamba_blocks):
      self.layers.append(MambaBlock(d_model, d_inner, d_state))
      if (i + 1) % attention_interval == 0 and (i + 1) < num_mamba_blocks:
        self.layers.append(GlobalSharedAttention(d_model))
        print(f"Added GlobalSharedAttention after MambaBlock {i+1}")

    self.final_proj = None
    if vocab_size is not None:
      self.final_proj = nn.Linear(d_model, vocab_size)

    print(f"Zamba model initialized with {num_mamba_blocks} Mamba blocks and attention every {attention_interval} blocks.")

  def forward(self, x):
    # Apply embedding layer to input token IDs
    x = self.embedding(x)

    for layer in self.layers:
      x = layer(x)

    if self.final_proj is not None:
      x = self.final_proj(x)

    return x