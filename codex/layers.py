import torch
import torch.nn as nn
import math

# STABLE MULTI-HEAD ATTENTION
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear projections
        Q = self.wq(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, heads, seq, seq)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Add mask (contains -1e9)
            attention_scores = attention_scores + mask

        # Subtract max for softmax stability (torch.softmax does this, but being explicit helps)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Safety net: Convert any NaN to 0 (happens if a row is entirely masked)
        attention_weights = torch.nan_to_num(attention_weights)

        context_layer = torch.matmul(attention_weights, V)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Fix: Return a tuple to match expected unpacking in TransformerBlock
        return self.wo(context_layer), None

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()

        # Re-initialize weights with Xavier uniform for stability
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0);
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0);

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = PositionWiseFeedForward(embed_dim, ff_dim)

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Pre-LayerNorm for Self-attention part
        normed_x = self.layernorm1(x)
        
        attn_output, _ = self.attn(normed_x, normed_x, normed_x, mask) # Pass mask to attention
        x = x + self.dropout1(attn_output) # Residual connection

        # Pre-LayerNorm for Feed-forward part
        normed_x = self.layernorm2(x)
        ffn_output = self.ffn(normed_x)
        x = x + self.dropout2(ffn_output) # Residual connection
        return x

# Token Embedding Layer
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialize embedding weights with smaller std for stability (or rely on default)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02);
        # Removed self.embed_dim and scaling from here

    def forward(self, tokens):
        # No scaling here; scaling will be applied in TransformerDecoder after positional encoding
        return self.embedding(tokens)

# Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
