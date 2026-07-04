import torch
import torch.nn as nn
from layers import TransformerBlock

class QwenModel(nn.Module):
  def __init__(self,vocab_size,embed_dim,num_layers,num_heads,ff_dim,max_seq_len,dropout=0.0):
    super().__init__()
    self.token_embedding=nn.Embedding(vocab_size,embed_dim)
    self.layers=nn.ModuleList([
        TransformerBlock(embed_dim,num_heads,ff_dim,dropout,use_rope=True)
        for _ in range(num_layers)
    ])
    self.norm_final=nn.LayerNorm(embed_dim)
    self.lm_head=nn.Linear(embed_dim,vocab_size,bias=False)
    self.max_seq_len=max_seq_len
  def forward(self,input_ids,attention_mask=None):
    batch_size,seq_len=input_ids.size()
    causal_mask=torch.tril(torch.ones((seq_len,seq_len),device=input_ids.device)).bool()
    if attention_mask is not None:
      causal_mask=causal_mask.unsqueeze(0).unsqueeze(0) & attention_mask.unsqueeze(1).unsqueeze(1)
    embeddings=self.token_embedding(input_ids)
    x=embeddings
    for layer in self.layers:
      x=layer(x,mask=causal_mask)
    x=self.norm_final(x)
    return self.lm_head(x)