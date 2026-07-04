import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import apply_rotary_pos_emb

class RMSNorm(nn.Module):
  def __init__(self,hidden_size,eps=1e-6):
    super().__init__()
    self.weight=nn.Parameter(torch.ones(hidden_size))
    self.eps=eps
  def forward(self,hidden_states):
    variance=hidden_states.to(torch.float32).pow(2).mean(-1,keepdim=True)
    hidden_states=hidden_states*torch.rsqrt(variance+self.eps)
    return self.weight*hidden_states.to(self.weight.dtype)

class RotaryEmbedding(nn.Module):
  def __init__(self,dim,max_position_embeddings=1024,base=10000.0):
    super().__init__()
    self.dim=dim
    self.max_position_embeddings=max_position_embeddings
    self.base=base
    inv_freq=1.0/(base**(torch.arange(0,dim,2).float()/dim))
    self.register_buffer('inv_freq',inv_freq)
    self._set_cos_sin_cache(seq_len=max_position_embeddings)
  def _set_cos_sin_cache(self,seq_len,device=None,dtype=None):
    self.max_seq_len_cached=seq_len
    t=torch.arange(seq_len,device=device,dtype=self.inv_freq.dtype)
    freqs=torch.outer(t,self.inv_freq)
    emb=torch.cat((freqs,freqs),dim=-1)
    self.register_buffer('cos_cached',emb.cos()[None,None,:,:].to(dtype),persistent=False)
    self.register_buffer('sin_cached',emb.sin()[None,None,:,:].to(dtype),persistent=False)
  def forward(self,seq_len,device,dtype):
    if seq_len>self.max_seq_len_cached:
      self._set_cos_sin_cache(seq_len=seq_len,device=device,dtype=dtype)
    return(
      self.cos_cached[:,:,:seq_len,:].to(device=device,dtype=dtype),
      self.sin_cached[:,:,:seq_len,:].to(device=device,dtype=dtype)
    )

class StarCoderV2Embeddings(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.token_embeddings=nn.Embedding(config.vocab_size,config.hidden_size)
    self.dropout=nn.Dropout(config.dropout_rate)
    self.config=config
  def forward(self,input_ids):
    token_embeddings=self.token_embeddings(input_ids)
    embeddings=self.dropout(token_embeddings)
    return embeddings

class StarCoderV2Attention(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.hidden_size=config.hidden_size
    self.num_attention_heads=config.num_attention_heads
    self.num_key_value_heads=config.num_key_value_heads
    self.num_key_value_groups=config.num_key_value_groups
    self.head_dim=self.hidden_size//self.num_attention_heads
    self.q_proj=nn.Linear(self.hidden_size,self.num_attention_heads*self.head_dim,bias=False)
    self.k_proj=nn.Linear(self.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
    self.v_proj=nn.Linear(self.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
    self.o_proj=nn.Linear(self.num_attention_heads*self.head_dim,self.hidden_size,bias=False)
    self.dropout=nn.Dropout(config.dropout_rate)
    self.rope=RotaryEmbedding(self.head_dim,max_position_embeddings=config.max_position_embeddings,base=config.rope_theta)
  def _split_heads(self,x,num_heads,batch_size):
    x=x.view(batch_size,-1,num_heads,self.head_dim)
    return x.permute(0,2,1,3)
  def _merge_heads(self,x,batch_size):
    x=x.permute(0,2,1,3)
    return x.reshape(batch_size,-1,self.num_attention_heads*self.head_dim)
  def forward(self,hidden_states,attention_mask=None,position_ids=None):
    batch_size,seq_len,_=hidden_states.size()
    query=self.q_proj(hidden_states)
    key=self.k_proj(hidden_states)
    value=self.v_proj(hidden_states)
    query=self._split_heads(query,self.num_attention_heads,batch_size)
    key=self._split_heads(key,self.num_key_value_heads,batch_size)
    value=self._split_heads(value,self.num_key_value_heads,batch_size)
    cos,sin=self.rope(seq_len,hidden_states.device,query.dtype)
    query,key=apply_rotary_pos_emb(query,key,cos,sin,position_ids)
    if self.num_key_value_groups>1:
      key=key.repeat_interleave(self.num_key_value_groups,dim=1)
      value=value.repeat_interleave(self.num_key_value_groups,dim=1)
    attn_scores=torch.matmul(query,key.transpose(-1,-2))/(self.head_dim**0.5)
    causal_mask=torch.triu(torch.ones(seq_len,seq_len,device=hidden_states.device,dtype=torch.bool),diagonal=1)
    attn_scores=attn_scores.masked_fill(causal_mask,-torch.inf)
    if attention_mask is not None:
      attn_scores=attn_scores+attention_mask
    attn_weights=F.softmax(attn_scores,dim=-1)
    attn_weights=self.dropout(attn_weights)
    attn_output=torch.matmul(attn_weights,value)
    attn_output=self._merge_heads(attn_output,batch_size)
    return self.o_proj(attn_output)

class StartCoderV2MLP(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.gate_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
    self.up_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
    self.down_proj=nn.Linear(config.intermediate_size,config.hidden_size,bias=False)
    self.act_fn=nn.SiLU()
    self.dropout=nn.Dropout(config.dropout_rate)
  def forward(self,hidden_states):
    gate_output=self.act_fn(self.gate_proj(hidden_states))
    up_output=self.up_proj(hidden_states)
    hidden_states=gate_output*up_output
    hidden_states=self.down_proj(hidden_states)
    return self.dropout(hidden_states)

class StarCoderV2Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.input_layernorm=RMSNorm(config.hidden_size,eps=config.norm_epsilon)
    self.self_attention=StarCoderV2Attention(config)
    self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.norm_epsilon)
    self.mlp=StartCoderV2MLP(config)
  def forward(self,hidden_states,attention_mask=None,position_ids=None):
    residual=hidden_states
    hidden_states=self.input_layernorm(hidden_states)
    attn_output=self.self_attention(hidden_states,attention_mask,position_ids)
    hidden_states=residual+attn_output
    residual=hidden_states
    hidden_states=self.post_attention_layernorm(hidden_states)
    mlp_output=self.mlp(hidden_states)
    hidden_states=residual+mlp_output
    return hidden_states
