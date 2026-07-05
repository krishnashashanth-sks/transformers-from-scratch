import torch
import torch.nn as nn

# Helper module for Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# 1. Patch Embedding Layer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolution to extract patches and project them to embed_dim
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

# 2. Multi-Head Self-Attention (MHSA)
class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch_size, n_patches + 1, embed_dim)

        batch_size, n_tokens, _ = x.shape

        # Generate Q, K, V matrices
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale # (batch_size, n_heads, n_tokens, n_tokens)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, self.embed_dim)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 3. Feed-Forward Network (MLP)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 4. Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4., dropout_rate=0.0):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, n_heads, dropout_rate)
        self.norm2 = LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) # Residual connection + MHSA
        x = x + self.mlp(self.norm2(x))  # Residual connection + MLP
        return x
