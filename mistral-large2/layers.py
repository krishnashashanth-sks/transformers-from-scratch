import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import apply_rotary_pos_emb
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
  def __init__(
      self,hidden_size:int,
      num_heads:int,num_kv_heads:int,
      head_dim:int,rope_freq_cis:torch.Tensor,
      max_seq_len:int=4096,
      window_size:int=0
  ):
    super().__init__()
    self.hidden_size=hidden_size
    self.num_heads=num_heads
    self.head_dim=head_dim
    self.num_kv_heads=num_kv_heads
    self.rope_freq_cis=rope_freq_cis
    self.max_seq_len=max_seq_len
    self.window_size=window_size
    self.wq=nn.Linear(hidden_size,num_heads*head_dim,bias=False)
    self.wk=nn.Linear(hidden_size,num_kv_heads*head_dim,bias=False)
    self.wv=nn.Linear(hidden_size,num_kv_heads*head_dim,bias=False)
    self.wo=nn.Linear(num_heads*head_dim,hidden_size,bias=False)
    self.attention_dropout=nn.Dropout(0.1)
    self.kv_heads_multiplier=self.num_heads//self.num_kv_heads

  def forward(self,x:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    bsz, seq_len, _ = x.shape

    xq = self.wq(x)
    xk = self.wk(x)
    xv = self.wv(x)

    xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

    xq, xk = apply_rotary_pos_emb(xq, xk, self.rope_freq_cis[:seq_len, :])

    # Grouped Query Attention (GQA) logic
    xk = xk.repeat_interleave(self.kv_heads_multiplier, dim=2)
    xv = xv.repeat_interleave(self.kv_heads_multiplier, dim=2)

    xq = xq.transpose(1,2)
    xk = xk.transpose(1,2)
    xv = xv.transpose(1,2)

    scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.head_dim)

    if mask is not None:
      scores = scores + mask

    if self.window_size > 0:
      # Sliding Window Attention mask
      # Create a causal mask for the window
      window_mask = torch.full((seq_len, seq_len), float('-inf'), device=scores.device, dtype=scores.dtype)
      for i in range(seq_len):
          start_idx = max(0, i - self.window_size + 1)
          window_mask[i, start_idx:i+1] = 0.0

      # Combine with causal mask from the input (if any)
      if mask is not None:
          combined_mask = mask + window_mask[None, None, :, :]
          scores = scores + combined_mask
      else:
          scores = scores + window_mask[None, None, :, :]

    attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = self.attention_dropout(attn_weights)
    output = torch.matmul(attn_weights, xv)
    output = output.transpose(1,2).contiguous().view(bsz,seq_len,-1)
    return self.wo(output)
  
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: (x @ gate_proj) * silu(x @ up_proj) @ down_proj
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

class TransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        head_dim: int, intermediate_size: int, rope_freq_cis: torch.Tensor,
        max_seq_len: int = 4096, window_size: int = 0
    ):
        super().__init__()
        self.attention = Attention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            rope_freq_cis, max_seq_len, window_size
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size)
        self.attention_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pre-attention normalization and attention with residual connection
        h = x + self.attention.forward(self.attention_norm(x), mask)
        # Pre-FFN normalization and FFN with residual connection
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
