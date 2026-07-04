import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rotary_pos_emb(x,cos,sin):
  x_rot=x[...,0::2]
  x_neg_rot=x[...,1::2]
  x_merged=torch.stack((-x_neg_rot,x_rot),dim=-1).flatten(-2)
  rotated_x=x*cos+x_merged*sin
  return rotated_x

class RotaryPositionalEmbedding(nn.Module):
  def __init__(self,dim,max_seq_len=2048,base=10000):
    super().__init__()
    inv_freq=1./(base** (torch.arange(0,dim,2).float()/dim))
    self.register_buffer('inv_freq',inv_freq)
    self.max_seq_len_cached=0 # Changed from None to 0
    self.cos_cached=None
    self.sin_cached=None
  def forward(self,x):
    seq_len=x.shape[1] 
    if seq_len>self.max_seq_len_cached:
      self.max_seq_len_cached=seq_len
      t=torch.arange(seq_len,device=x.device,dtype=self.inv_freq.dtype)
      freqs=torch.einsum('i,j->ij',t,self.inv_freq)
      emb=torch.cat((freqs,freqs),dim=-1)
      self.cos_cached=emb.cos()[:,None,:]
      self.sin_cached=emb.sin()[:,None,:]
    return self.cos_cached[:seq_len,...],self.sin_cached[:seq_len,...]
  
class MultiHeadSelfAttention(nn.Module):
  def __init__(self,embed_dim,num_heads,dropout=0.0,use_rope=True):
    super().__init__()
    self.embed_dim=embed_dim
    self.num_heads=num_heads
    self.head_dim=embed_dim//num_heads
    assert self.head_dim*num_heads==self.embed_dim,'embed_dim must be divisible by num_heads'
    self.q_proj=nn.Linear(embed_dim,embed_dim,bias=False)
    self.k_proj=nn.Linear(embed_dim,embed_dim,bias=False)
    self.v_proj=nn.Linear(embed_dim,embed_dim,bias=False)
    self.out_proj=nn.Linear(embed_dim,embed_dim,bias=False)
    self.dropout=nn.Dropout(dropout)
    self.use_rope=use_rope
    if self.use_rope:
      self.rope=RotaryPositionalEmbedding(self.head_dim)
  def forward(self,x,mask=None):
    batch_size,seq_len,_=x.size()
    q=self.q_proj(x).view(batch_size,seq_len,self.num_heads,self.head_dim)
    k=self.k_proj(x).view(batch_size,seq_len,self.num_heads,self.head_dim)
    v=self.v_proj(x).view(batch_size,seq_len,self.num_heads,self.head_dim)
    if self.use_rope:
      cos,sin=self.rope(q)
      q=apply_rotary_pos_emb(q,cos,sin)
      k=apply_rotary_pos_emb(k,cos,sin)
    q=q.transpose(1,2)
    k=k.transpose(1,2)
    v=v.transpose(1,2)
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
    if mask is not None:
      scores=scores.masked_fill(mask==0,float('-inf'))
    attention_weights=F.softmax(scores,dim=-1)
    attention_weights=self.dropout(attention_weights)
    context=torch.matmul(attention_weights,v)
    context=context.transpose(1,2).contiguous().view(batch_size,seq_len,self.embed_dim)
    return self.out_proj(context)
  
class FeedForward(nn.Module):
  def __init__(self,embed_dim,ff_dim,dropout=0.0):
    super().__init__()
    self.linear1=nn.Linear(embed_dim,ff_dim)
    self.gelu=nn.GELU()
    self.linear2=nn.Linear(ff_dim,embed_dim)
    self.dropout=nn.Dropout(dropout)
  def forward(self,x):
    return self.dropout(self.linear2(self.gelu(self.linear1(x))))
  
class TransformerBlock(nn.Module):
  def __init__(self,embed_dim,num_heads,ff_dim,dropout=0.0,use_rope=True):
    super().__init__()
    self.norm1=nn.LayerNorm(embed_dim)
    self.attn=MultiHeadSelfAttention(embed_dim,num_heads,dropout,use_rope)
    self.ffn=FeedForward(embed_dim,ff_dim,dropout)
    self.norm2=nn.LayerNorm(embed_dim)
  def forward(self,x,mask=None):
    norm_x=self.norm1(x)
    attn_output=self.attn(norm_x,mask=mask)
    x=x+attn_output

    norm_x=self.norm2(x)
    ffn_output=self.ffn(norm_x)
    x=x+ffn_output
    return x