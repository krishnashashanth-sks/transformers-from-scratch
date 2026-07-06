import torch
import torch.nn as nn
from layers import TokenEmbedding,TransformerBlock,RMSNorm
from utils import precompute_freqs_cis

class MistralLarge2(nn.Module):
  def __init__(self,vocab_size:int,num_layers:int,hidden_size:int,
               num_heads:int,num_kv_heads:int,intermediate_size:int,
               max_seq_len:int=4096,window_size:int=0,rope_theta:float=1000.0):
    super().__init__()
    self.vocab_size=vocab_size
    self.num_layers=num_layers
    self.hidden_size=hidden_size
    self.num_heads=num_heads
    self.num_kv_heads=num_kv_heads
    self.intermediate_size=intermediate_size
    self.max_seq_len=max_seq_len
    self.rope_theta=rope_theta
    self.embed_tokens=TokenEmbedding(vocab_size,hidden_size)
    head_dim=hidden_size//num_heads
    self.rope_freq_cls=precompute_freqs_cis(
        head_dim,max_seq_len*2,self.rope_theta
    )
    self.layers=nn.ModuleList([
        TransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rope_freq_cis=self.rope_freq_cls,
            max_seq_len=max_seq_len,
            window_size=window_size
        )
        for _ in range(num_layers)
    ])
    self.norm=RMSNorm(hidden_size)
    self.output=nn.Linear(hidden_size,vocab_size,bias=False)

  def forward(self,input_ids):
    batch_size,seq_len=input_ids.shape
    assert seq_len<=self.max_seq_len,f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
    h=self.embed_tokens(input_ids)
    # Mask should be float type for -inf values
    mask=torch.full((1,1,seq_len,seq_len),float('-inf'),device=input_ids.device,dtype=torch.float32)
    mask=torch.triu(mask,diagonal=1)
    for layer in self.layers:
      h=layer(h,mask)
    return self.output(self.norm(h))