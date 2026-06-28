import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
  def __init__(self,dim,eps=1e-6):
    super().__init__()
    self.eps=eps
    self.weight=nn.Parameter(torch.ones(dim))
  def _norm(self,x):
    return x* torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
  def forward(self,x):
    return self._norm(x)*self.weight
  
class RotaryPositionalEmbedding(nn.Module):
  def __init__(self,dim,seq_len_interpolation_factor=1.0):
    super().__init__()
    self.dim=dim
    self.seq_len_interpolation_factor=seq_len_interpolation_factor
    inv_freq=1.0/(10000**(torch.arange(0,dim,2).float()/dim))
    self.register_buffer('inv_freq',inv_freq)
  @staticmethod
  def rotate_half(x):
    x1,x2=x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)
  def forward(self,x,offset=0):
    seq_len=x.shape[1]
    effective_seq_len=seq_len/self.seq_len_interpolation_factor
    position_ids=torch.arange(seq_len,device=x.device,dtype=self.inv_freq.dtype)+offset
    if self.seq_len_interpolation_factor!=1.0:
      position_ids=position_ids/self.seq_len_interpolation_factor
    t=torch.einsum('i,j->ij',position_ids,self.inv_freq)
    cos_t=t.cos()
    sin_t=t.sin()
    cos_t=cos_t.repeat(1,2).view(1,seq_len,self.dim)
    sin_t=sin_t.repeat(1,2).view(1,seq_len,self.dim)
    # Apply the rotation formula
    # x * cos(t) + rotate_half(x) * sin(t)
    return (x * cos_t) + (self.rotate_half(x)*sin_t)
  
class MultiHeadAttention(nn.Module):
  def __init__(self,dim,num_heads,dropout_rate,rope_layer):
    super().__init__()
    self.dim=dim
    self.num_heads=num_heads
    self.head_dim=dim//num_heads
    assert dim%num_heads==0,'dim must be divisible by num_heads'
    self.q_proj=nn.Linear(dim,dim,bias=False)
    self.k_proj=nn.Linear(dim,dim,bias=False)
    self.v_proj=nn.Linear(dim,dim,bias=False)
    self.out_proj=nn.Linear(dim,dim,bias=False)
    self.dropout=nn.Dropout(dropout_rate)
    self.rope_layer=rope_layer
  def forward(self,x,mask=None):
    batch_size,seq_len,_=x.shape
    q=self.q_proj(x)
    k=self.k_proj(x)
    v=self.v_proj(x)
    q=q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
    k=k.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
    v=v.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

    # Apply RoPE
    q_rotated = self.rope_layer(q.reshape(-1, seq_len, self.head_dim))
    k_rotated = self.rope_layer(k.reshape(-1, seq_len, self.head_dim))

    # Reshape back to (batch_size, num_heads, seq_len, head_dim)
    q = q_rotated.view(batch_size, self.num_heads, seq_len, self.head_dim)
    k = k_rotated.view(batch_size, self.num_heads, seq_len, self.head_dim)

    attn_scores=(q @ k.transpose(-2,-1))/math.sqrt(self.head_dim)
    if mask is not None:
      # `mask==0` correctly identifies masked positions for boolean masks (False for masked)
      attn_scores=attn_scores.masked_fill(mask==0,float('-1e9'))
    attn_probs=torch.softmax(attn_scores,dim=-1)
    attn_probs=self.dropout(attn_probs)
    attn_output=attn_probs @ v
    attn_output=attn_output.transpose(1,2).contiguous().view(batch_size,seq_len,self.dim)
    final_output=self.out_proj(attn_output) # Fixed typo: changed 'att_output' to 'final_output'
    return final_output # Return the correctly named output variable
  
class SwiGLUFeedForward(nn.Module):
  def __init__(self,dim,hidden_dim,dropout_rate):
    super().__init__()
    self.gate_proj=nn.Linear(dim,hidden_dim,bias=False)
    self.up_proj=nn.Linear(dim,hidden_dim,bias=False)
    self.down_proj=nn.Linear(hidden_dim,dim,bias=False)
    self.dropout=nn.Dropout(dropout_rate)
  def forward(self,x):
    gate=self.gate_proj(x)
    up=self.up_proj(x)
    hidden_states=F.silu(gate)*up
    output=self.down_proj(hidden_states)
    return self.dropout(output)
  
class LlamaDecoderBlock(nn.Module):
  def __init__(self,dim,num_heads,hidden_dim,dropout_rate,rope_layer):
    super().__init__()
    self.attention_norm=RMSNorm(dim)
    self.attn=MultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        rope_layer=rope_layer
    )
    self.ffn_norm=RMSNorm(dim)
    self.ffn=SwiGLUFeedForward(
        dim=dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )
  def forward(self,x,mask=None):
    h=self.attention_norm(x)
    h=self.attn(h,mask=mask)
    x=x+h
    h=self.ffn_norm(x)
    h=self.ffn(h)
    return x+h