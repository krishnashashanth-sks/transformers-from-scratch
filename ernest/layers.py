import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x): # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(0.1) # Standard dropout rate for Transformers

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_model)
        if mask is not None:
            mask = mask.unsqueeze(1) # (batch_size, 1, 1, seq_len) for broadcasting

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => num_heads x d_k
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # query, key, value: (batch_size, num_heads, seq_len, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Fill with large negative for softmax
        p_attn = self.dropout(nn.functional.softmax(scores, dim=-1))

        # 3) "Concat" using a view and apply a final linear.
        # x: (batch_size, num_heads, seq_len, d_k)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.linears[-1](x) # Final linear projection

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))

class TransformerEncoderLayer(nn.Module):
    """Comprises self-attention and feed-forward networks."""
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

class TransformerEncoder(nn.Module):
    """A stack of N TransformerEncoderLayers."""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_seq_len=100):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) # Final layer norm

        nn.init.xavier_uniform_(self.token_embedding.weight)

    def forward(self, src, src_mask=None):
        # src: (batch_size, seq_len)
        # src_mask: (batch_size, 1, seq_len) or (batch_size, seq_len, seq_len)

        # (batch_size, seq_len, d_model)
        x = self.token_embedding(src)
        # Add positional encoding (operates on seq_len, batch_size, d_model, so need transpose)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1) # (batch_size, seq_len, d_model)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x) # Apply final layer norm