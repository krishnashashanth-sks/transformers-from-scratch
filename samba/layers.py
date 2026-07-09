import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaLayer(nn.Module):
  def __init__(self,d_model:int,d_state:int=16,d_inner:int=None,
               conv_kernel_size:int=4,bias:bool=False):
    super().__init__()
    self.d_model=d_model
    self.d_state=d_state
    self.d_inner=d_inner if d_inner is not None else d_model * 2
    self.conv_kernel_size=conv_kernel_size
    self.in_proj=nn.Linear(d_model,2*self.d_inner,bias=bias)
    self.conv1d=nn.Conv1d(
        in_channels=self.d_inner,
        out_channels=self.d_inner,
        kernel_size=conv_kernel_size,
        groups=self.d_inner,
        padding=conv_kernel_size-1,
        bias=bias
    )
    self.x_to_ssm_params=nn.Linear(self.d_inner,self.d_model+2*self.d_model*self.d_state,bias=False)
    self.ssm_input_proj=nn.Linear(self.d_inner,self.d_model,bias=False)
    self.out_proj=nn.Linear(self.d_model,d_model,bias=bias)
    self.gate_proj=nn.Linear(self.d_inner,self.d_model,bias=bias)
    self.A=nn.Parameter(torch.arange(1,d_state+1).float().repeat(d_model,1))
    self.D=nn.Parameter(torch.ones(d_model))
  def _selective_scan(self,u,A_bar,B_bar,C):
    batch_size,seq_len,d_model=u.shape
    h=torch.zeros(batch_size,d_model,self.d_state,device=u.device)
    outputs=[]
    for i in range(seq_len):
      h=A_bar[:,i]*h+B_bar[:,i]*u[:,i].unsqueeze(-1)
      output_i=torch.sum(h*C[:,i],dim=-1)+self.D*u[:,i]
      outputs.append(output_i)
    return torch.stack(outputs,dim=1)
  def forward(self,x):
    batch_size,seq_len,_=x.shape 
    x_and_z=self.in_proj(x)
    x_ssm_fork,z=x_and_z.chunk(2,dim=-1) 
    x_ssm_conv_input = x_ssm_fork.transpose(1,2) 
    x_ssm_conv_pre_act=self.conv1d(x_ssm_conv_input)
    x_ssm_act=F.silu(x_ssm_conv_pre_act)[:,:,:seq_len]
    x_ssm_conv_out=x_ssm_act.transpose(1,2)
    ssm_params=self.x_to_ssm_params(x_ssm_conv_out)
    dt,B_flat,C_flat=ssm_params.split(
        [self.d_model,self.d_model*self.d_state,self.d_model*self.d_state],
        dim=-1
    )
    B=B_flat.view(batch_size,seq_len,self.d_model,self.d_state) 
    C=C_flat.view(batch_size,seq_len,self.d_model,self.d_state) 
    dt=F.softplus(dt)
    dt_reshaped=dt.unsqueeze(-1) 
    A_broadcastable=self.A.float().unsqueeze(0).unsqueeze(0)
    A_bar=torch.exp(dt_reshaped*A_broadcastable)
    B_bar=dt_reshaped*B
    u=self.ssm_input_proj(x_ssm_conv_out)
    ssm_output=self._selective_scan(u,A_bar,B_bar,C)
    out_proj_output=self.out_proj(ssm_output)
    return out_proj_output*F.silu(self.gate_proj(z)) 
  
class SlidingWindowAttention(nn.Module):
  def __init__(self,d_model:int,num_heads:int,window_size:int):
    super().__init__()
    if d_model%num_heads!=0:
      raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    self.d_model=d_model
    self.num_heads=num_heads
    self.window_size=window_size
    self.head_dim=d_model//num_heads
    self.query_proj=nn.Linear(d_model,d_model)
    self.key_proj=nn.Linear(d_model,d_model)
    self.value_proj=nn.Linear(d_model,d_model)
    self.out_proj=nn.Linear(d_model,d_model)
    print(f"SlidingWindowAttention initialized with d_model={d_model}, num_heads={num_heads}, window_size={window_size}")
  def forward(self,x):
    batch_size,seq_len,_=x.shape
    query=self.query_proj(x)
    key=self.key_proj(x)
    value=self.value_proj(x)
    query=query.view(batch_size,seq_len,self.num_heads,self.head_dim)
    key=key.view(batch_size,seq_len,self.num_heads,self.head_dim)
    value=value.view(batch_size,seq_len,self.num_heads,self.head_dim)
    query=query.transpose(1,2)
    key=key.transpose(1,2)
    value=value.transpose(1,2)
    output=torch.zeros_like(query,device=x.device)
    for i in range(seq_len):
      start_idx=max(0,i-self.window_size+1)
      end_idx=i+1
      q_i=query[:,:,i:i+1,:]
      k_window=key[:,:,start_idx:end_idx,:]
      attention_scores=(q_i @ k_window.transpose(-2,-1))/(self.head_dim**0.5)
      attention_weights=F.softmax(attention_scores,dim=-1)
      attended_value=attention_weights @ k_window
      output[:,:,i:i+1,:]=attended_value
    output=output.transpose(1,2).contiguous()
    output=output.view(batch_size,seq_len,self.d_model)
    final_output=self.out_proj(output)
    return final_output
  
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_inner: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner if d_inner is not None else d_model * 2

        # Initialize three linear layers
        self.w1 = nn.Linear(self.d_model, self.d_inner) # Maps d_model to d_inner
        self.w2 = nn.Linear(self.d_model, self.d_inner) # Used for gating mechanism
        self.w3 = nn.Linear(self.d_inner, self.d_model) # Maps d_inner back to d_model


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply w1 to x
        x_w1 = self.w1(x)

        # Apply w2 to x and then SiLU activation
        x_w2_activated = F.silu(self.w2(x))

        # Multiply the results
        gated_output = x_w1 * x_w2_activated

        # Apply w3 to the product
        final_output = self.w3(gated_output)

        return final_output

class SambaBlock(nn.Module):
  def __init__(self,d_model,d_state,num_heads,window_size,d_inner_mamba=None,conv_kernel_size=None,d_inner_swiglu=None,bias=False):
    super().__init__()
    self.d_model=d_model
    self.norm1=nn.LayerNorm(d_model)
    self.norm2=nn.LayerNorm(d_model)
    self.norm3=nn.LayerNorm(d_model)
    self.mamba=MambaLayer(
        d_model=d_model,
        d_state=d_state,
        d_inner=d_inner_mamba,
        conv_kernel_size=conv_kernel_size,
        bias=bias
    )
    self.attention=SlidingWindowAttention(
        d_model=d_model,
        num_heads=num_heads,
        window_size=window_size
    )
    self.feed_forward=SwiGLU(
        d_model=d_model,
        d_inner=d_inner_swiglu
    )
  def forward(self,x):
    x_normed=self.norm1(x)
    mamba_output=self.mamba(x_normed)
    x=x+mamba_output
    x_normed=self.norm2(x)
    attention_output=self.attention(x_normed)
    x=x+attention_output
    x_normed=self.norm3(x)
    feed_forward_output=self.feed_forward(x)
    x=x+feed_forward_output
    return x