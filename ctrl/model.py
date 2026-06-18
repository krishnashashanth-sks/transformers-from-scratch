import torch.nn as nn
import torch
from layers import Embeddings
from layers import TransformerDecoderBlock

class BasicCTRLModel(nn.Module):
  def __init__(self,vocab_size,d_model,max_seq_len,n_heads,d_ff,n_layers,dropout_rate):
    super(BasicCTRLModel,self).__init__()
    self.embeddings=Embeddings(vocab_size,d_model,max_seq_len,dropout_rate)
    self.decoder_layers=nn.ModuleList([
        TransformerDecoderBlock(d_model,n_heads,d_ff,dropout_rate)
        for _ in range(n_layers)
    ])
    self.lm_head=nn.Linear(d_model,vocab_size)
    self.dropout=nn.Dropout(dropout_rate)
  def forward(self,input_ids,attention_mask=None):
    x=self.embeddings(input_ids)
    seq_len=input_ids.size(1)
    causal_mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1).type(torch.uint8)==0
    causal_mask=causal_mask.to(input_ids.device).unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
      attention_mask_expanded=attention_mask.unsqueeze(1).unsqueeze(1)
      mask=causal_mask & attention_mask_expanded
    else:
      mask=causal_mask
    for block in self.decoder_layers:
      x=block(x,mask)
    return self.lm_head(x)