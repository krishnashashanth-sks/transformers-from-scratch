import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask for decoder-only architecture (causal masking)
        if mask is not None:
            # Ensure mask is broadcastable to (batch_size, num_heads, seq_len, seq_len)
            # Create a causal mask if not provided or combine with attention_mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            # Combine attention_mask (if it's not None) with causal_mask
            if mask.dim() == 2: # attention_mask is typically (batch_size, seq_len)
                expanded_mask = mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
                attn_mask = expanded_mask & causal_mask # (batch_size, 1, seq_len, seq_len)
            else: # Assume mask is already (seq_len, seq_len) or (1, 1, seq_len, seq_len)
                attn_mask = mask & causal_mask

            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        else: # If no mask is provided, still apply causal mask for decoder
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and project back to original embedding dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(), # GPT models typically use GELU activation
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        # LayerNorm before attention, then attention, then dropout, then residual
        attn_output = self.attention(self.norm1(x), mask=attention_mask)
        x = x + self.dropout1(attn_output) # Residual connection

        # LayerNorm before feed-forward, then feed-forward, then dropout, then residual
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output) # Residual connection
        return x
