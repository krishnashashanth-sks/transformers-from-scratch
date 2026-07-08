import torch.nn as nn
import torch
import math

# 1. MultiHeadSelfAttention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.dense(output)
        return output

# 2. PositionwiseFeedForward 
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

# 3. EncoderBlock 
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

# 4. ConceptualMultilingualEmbedding 
class ConceptualMultilingualEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.token_embedding(token_ids)


# 5. ConceptualMultiModalInput 
class ConceptualMultiModalInput(nn.Module):
    def __init__(self, text_vocab_size, text_embed_dim, image_feature_dim, common_embed_dim, max_seq_len):
        super().__init__()
        self.text_embedding = ConceptualMultilingualEmbedding(text_vocab_size, text_embed_dim) # Use our defined embedder
        self.text_proj = nn.Linear(text_embed_dim, common_embed_dim)
        self.image_proj = nn.Linear(image_feature_dim, common_embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, common_embed_dim)

    def forward(self, text_tokens, image_features):
        text_embed = self.text_embedding(text_tokens)
        text_proj_embed = self.text_proj(text_embed)
        image_proj_embed = self.image_proj(image_features)
        combined_embed = torch.cat((text_proj_embed, image_proj_embed), dim=1)
        seq_len = combined_embed.size(1)
        positions = torch.arange(seq_len, device=combined_embed.device).unsqueeze(0)
        pos_embed = self.position_embedding(positions)
        return combined_embed + pos_embed
