import torch
import torch.nn as nn
import torch.nn.functional as F

class HyenaOperator(nn.Module):
  def __init__(self,dim,order=2,seq_len=1024,filter_length=64):
    super().__init__()
    self.dim=dim
    self.order=order
    self.seq_len=seq_len
    self.filter_length=filter_length
    self.in_proj=nn.Linear(dim,dim*(order+1))
    self.learned_filter_coeffs=nn.Parameter(torch.randn(order,dim,filter_length)*0.02)
    self.decay_factors=nn.Parameter(torch.ones(order,dim)*0.9)
  def _synthesis_long_filter(self,order_idx,target_seq_len,device,dtype):
    filter_coeffs=self.learned_filter_coeffs[order_idx]
    decay_factor=self.decay_factors[order_idx].unsqueeze(-1)
    indices=torch.arange(self.filter_length,device=device,dtype=dtype)
    decay_curve=torch.pow(decay_factor,indices)
    base_filter=filter_coeffs*decay_curve
    long_filter=F.pad(base_filter,(0,target_seq_len-self.filter_length))
    return long_filter
  def forward(self,x):
    B,L,D=x.shape
    projected=self.in_proj(x)
    main_input=projected[...,:D]
    gates=[projected[...,D*(i+1):D*(i+2)]for i in range(self.order)]
    fft_len=2**(L-1).bit_length() if L>0 else 1
    hidden_state=main_input
    for i in range(self.order):
      current_gate=gates[i]
      h_long_time_domain=self._synthesis_long_filter(i,fft_len,x.device,x.dtype)
      h_long_freq_domain=torch.fft.rfft(h_long_time_domain,n=fft_len)
      x_conv_input=hidden_state.transpose(1,2)
      x_padded=F.pad(x_conv_input,(0,fft_len-L))
      x_freq_domain=torch.fft.rfft(x_padded,n=fft_len)
      convolved_freq_domain=x_freq_domain*h_long_freq_domain.unsqueeze(0)
      convolved_output_padded=torch.fft.irfft(convolved_freq_domain,n=fft_len)
      convolved_output=convolved_output_padded[...,:L].transpose(1,2)
      hidden_state=current_gate*convolved_output
    return hidden_state
class HyenaLayer(nn.Module):
  def __init__(self,dim,order=2,seq_len=1024,filter_length=64,dropout=0.1):
    super().__init__()
    self.norm=nn.LayerNorm(dim)
    self.hyena_op=HyenaOperator(dim,order,seq_len,filter_length)
    self.dropout=nn.Dropout(dropout)
    self.feed_forward=nn.Sequential(
        nn.Linear(dim,4*dim),
        nn.GELU(),
        nn.Linear(4*dim,dim),
        nn.Dropout(dropout)
    )
  def forward(self,x):
    norm_x=self.norm(x)
    hyena_out=self.hyena_op(norm_x)
    hyena_out=self.dropout(hyena_out)
    x=x+hyena_out
    norm_x=self.norm(x)
    ffn_output=self.feed_forward(norm_x)
    x=x+ffn_output
    return x
  
class PositionalEncoding(nn.Module):
  def __init__(self,d_model,dropout=0.1,max_len=5000):
    super().__init__()
    self.dropout=nn.Dropout(p=dropout)
    self.pe=nn.Parameter(torch.randn(1,max_len,d_model))
  def forward(self,x):
    x=x+self.pe[:,:x.size(1),:]
    return self.dropout(x)