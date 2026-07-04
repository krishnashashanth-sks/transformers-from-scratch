import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self,hidden_size,num_heads,intermediate_size,dropout_rate):
    super().__init__()
    self.attn=nn.MultiheadAttention(hidden_size,num_heads,dropout=dropout_rate,batch_first=True)
    self.norm1=nn.LayerNorm(hidden_size)
    self.dropout1=nn.Dropout(dropout_rate)
    self.ffn=nn.Sequential(
        nn.Linear(hidden_size,intermediate_size),
        nn.GELU(),
        nn.Linear(intermediate_size,hidden_size)
    )
    self.norm2=nn.LayerNorm(hidden_size)
    self.dropout2=nn.Dropout(dropout_rate)
  def forward(self,x,attention_mask=None, causal_mask=None):
    normed_x=self.norm1(x)
    key_padding_mask=(attention_mask==0) if attention_mask is not None else None
    attn_output,_=self.attn(query=normed_x,
                            key=normed_x,
                            value=normed_x,
                            key_padding_mask=key_padding_mask,
                            is_causal=True, # Keep is_causal=True as per original intention
                            attn_mask=causal_mask) # Pass the causal_mask here
    x=x+self.dropout1(attn_output)
    normed_x=self.norm2(x)
    return x+self.dropout2(self.ffn(normed_x))