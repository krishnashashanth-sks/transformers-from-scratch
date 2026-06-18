import torch.nn as nn 
import torch

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout_rate):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        seq_len = x.size(1)
        # Create positional indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # Combine token and positional embeddings
        embeddings = self.token_embeddings(x) + self.position_embeddings(positions)
        return self.dropout(embeddings)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Apply linear projections and split into heads
        query = self.query_proj(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 3) Apply mask if provided (for decoder, causality)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Fill masked positions with a very small number

        # 4) Apply softmax to get attention probabilities
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # 5) Multiply by values and concatenate heads
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        # 6) Apply final linear projection
        return self.out_proj(x)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout_rate)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # LayerNorm before attention (pre-norm style, common in modern transformers)
        # Self-Attention + Residual + Dropout + LayerNorm
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output) # Add & Norm
        x = self.norm1(x)

        # Feed-Forward + Residual + Dropout + LayerNorm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output) # Add & Norm
        x = self.norm2(x)
        return x