import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute the rotary embedding matrix
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Different frequencies for sin and cos components
        emb = torch.cat((freqs, freqs), dim=-1) # (max_seq_len, dim)
        # Unsqueeze for broadcasting across batch_size and num_heads
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0)) # (1, 1, max_seq_len, dim)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0)) # (1, 1, max_seq_len, dim)

    def forward(self, x):
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        seq_len = x.shape[-2]
        # Slice the precomputed values to the current sequence length
        cos = self.cos_cached[:, :, :seq_len, :].to(x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(x.device)

        # Apply rotation
        # Equivalent to x_j * cos(theta_j) - x_j+1 * sin(theta_j)
        # and x_j+1 * cos(theta_j) + x_j * sin(theta_j)
        # which is multiplication by a complex number (cos(theta) + i*sin(theta))
        x_rotated = x * cos + self._rotate_half(x) * sin
        return x_rotated

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, dropout_rate=0.0, max_seq_len=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # If num_kv_heads is not provided, it defaults to Multi-Head Attention (MHA)
        # If num_kv_heads=1, it acts as Multi-Query Attention (MQA)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        # The head_dim must be identical for Q, K, and V
        self.head_dim = embed_dim // num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        # Projections: Q uses all heads, K and V use a reduced set of heads
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        # Rotary Positional Embedding (Assumed pre-defined)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 1. Project and reshape to (B, L, H, D) -> (B, H, L, D)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Apply RoPE to Queries and Keys
        q = self.rope(q)
        k = self.rope(k)

        # 3. Grouped Attention: Repeat KV heads to match Q heads
        if self.num_groups > 1:
            # repeat_interleave ensures that heads [0,0,0, 1,1,1...] are grouped together
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # 4. Scaled Dot-Product Attention
        # Shape: (B, H, L, L)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # If mask is (batch, seq_len), unsqueeze to (batch, 1, 1, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            # If mask is (batch, 1, seq_len), unsqueeze to (batch, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Contextual output
        # Shape: (B, H, L, D)
        attn_output = torch.matmul(attn_weights, v)

        # 6. Reshape and Project back to embed_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)
    
class SwiGLU(nn.Module):
  def __init__(self,in_features,hidden_features):
    super().__init__()
    self.in_features=in_features
    self.hidden_features=hidden_features
    self.w1=nn.Linear(in_features,2*hidden_features,bias=False)
    self.w2=nn.Linear(hidden_features,in_features,bias=False)
  def forward(self,x):
    combined_output=self.w1(x)
    x1,x2=combined_output.chunk(2,dim=-1)
    activated_x2=F.silu(x2)
    gated_output=x2*activated_x2
    final_output=self.w2(gated_output)
    return final_output
  
class SwiGLUFeedForward(nn.Module):
  def __init__(self,embed_dim,hidden_dim,dropout_rate=0.0):
    super().__init__()
    self.embed_dim=embed_dim
    self.hidden_dim=hidden_dim
    # Corrected `in_features` from undefined variable to `embed_dim`
    # Corrected `hidden_features` argument to `hidden_dim` as expected by SwiGLU
    self.swiglu=SwiGLU(in_features=embed_dim, hidden_features=hidden_dim)
    self.dropout=nn.Dropout(dropout_rate)
  def forward(self,x):
    output=self.swiglu(x)
    return self.dropout(output)
  
class RMSNorm(nn.Module):
  def __init__(self,dim,eps=1e-6):
    super().__init__()
    self.eps=eps
    self.weight=nn.Parameter(torch.ones(dim))
  def _norm(self,x):
    return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
  def forward(self,x):
    return self.weight*self._norm(x)
class TransformerDecoderBlock(nn.Module):
  def __init__(self,embed_dim,num_heads,num_kv_heads,ffn_hidden_dim,dropout_rate,max_seq_len=2048):
    super().__init__()
    self.embed_dim=embed_dim
    # Pre-attention normalization
    self.attn_norm = RMSNorm(embed_dim)
    # Grouped-Query Attention layer
    self.attn = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout_rate=dropout_rate,
        max_seq_len=max_seq_len
    )

    # Pre-FFN normalization
    self.ffn_norm = RMSNorm(embed_dim)
    # SwiGLU Feed-Forward Network
    self.ffn = SwiGLUFeedForward(
        embed_dim=embed_dim,
        hidden_dim=ffn_hidden_dim,
        dropout_rate=dropout_rate
    )

    self.dropout = nn.Dropout(dropout_rate)
  def forward(self, x, attention_mask=None):
        # 1. Self-Attention Block
        norm_x = self.attn_norm(x)
        attn_out = self.attn(norm_x, mask=attention_mask)
        # FIX: Change 'x += self.dropout(...)' to 'x = x + self.dropout(...)'
        x = x + self.dropout(attn_out)

        # 2. Feed-Forward Block
        norm_x = self.ffn_norm(x)
        ffn_out = self.ffn(norm_x)
        # FIX: Change 'x += self.dropout(...)' to 'x = x + self.dropout(...)'
        x = x + self.dropout(ffn_out)

        return x