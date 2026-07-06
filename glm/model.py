import torch.nn as nn
from layers import PositionalEncoding,TransformerBlock,LayerNorm
import torch

class GLM5Model(nn.Module):
  def __init__(self,vocab_size,embed_dim,num_heads,ff_dim,num_layers,max_len=512,dropout=0.1):
    super().__init__()
    self.token_embedding=nn.Embedding(vocab_size,embed_dim)
    self.positional_encoding=PositionalEncoding(embed_dim,max_len)
    self.transformer_blocks=nn.ModuleList([
        TransformerBlock(embed_dim,num_heads,ff_dim,dropout)for _ in range(num_layers)
    ])
    self.norm=LayerNorm(embed_dim)
    self.output_layer=nn.Linear(embed_dim,vocab_size)
    self.dropout=nn.Dropout(dropout)
  def forward(self,x,mask=None):
    # Corrected: Use torch.tensor() instead of torch.Tensor() for scalar with dtype
    x = self.token_embedding(x) * torch.sqrt(torch.tensor(self.token_embedding.embedding_dim, dtype=torch.float32))
    x=self.positional_encoding(x)
    x=self.dropout(x)
    for block in self.transformer_blocks:
      x=block(x,mask)
    x=self.norm(x)
    return self.output_layer(x)