import torch.nn as nn
import torch
from layers import PositionalEncoding,HyenaLayer

class HyenaModel(nn.Module):
  def __init__(self,num_layers,dim,vocab_size,order=2,seq_len=1024,filter_length=64,dropout=0.1):
    super().__init__()
    self.embedding=nn.Embedding(vocab_size,dim)
    self.pos_encoder=PositionalEncoding(dim,dropout,max_len=seq_len)
    self.layers=nn.ModuleList([
        HyenaLayer(dim,order,seq_len,filter_length,dropout)
        for _ in range(num_layers)
    ])
    self.norm_output=nn.LayerNorm(dim)
    self.output_layer=nn.Linear(dim,vocab_size)
  def forward(self,src):
    x=self.embedding(src)*torch.sqrt(torch.tensor(self.embedding.embedding_dim,dtype=torch.float32))
    x=self.pos_encoder(x)
    for layer in self.layers:
      x=layer(x)
    return self.output_layer(x)