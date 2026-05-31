import torch
import torch.nn as nn

# Positional Encoding injects information about the relative or absolute position of tokens in the sequence.
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

  # Multi-Head Self-Attention is the core component of the Transformer, allowing the model to weigh the importance of different words in the input sequence when processing each word.
  class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Score
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        concat_output = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.wo(concat_output)
        return output

# A Transformer block consists of a multi-head self-attention layer, followed by a feed-forward neural network, with residual connections and layer normalization around each sub-layer.
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention sub-layer
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output)) # Add & Norm

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output)) # Add & Norm
        return x
# Basic GPT model with an embedding layer, positional encoding, a stack of Transformer blocks, and a final linear layer for token prediction.
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, num_layers, hidden_dim):
        super(SimpleGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, mask=None):
        # input_ids: (batch_size, seq_len)
        token_embeddings = self.token_embedding(input_ids)
        x = self.positional_encoding(token_embeddings)

        for block in self.transformer_blocks:
            x = block(x, mask)

        logits = self.output_layer(x)
        return logits 
