import torch.nn as nn
from einops import rearrange
from utils import pair
import torch

# ---  Patch Embedding Layer ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, image_size: int):
        super().__init__()
        self.patch_size = pair(patch_size)
        self.image_size = pair(image_size)

        assert (self.image_size[0] % self.patch_size[0] == 0) and \
               (self.image_size[1] % self.patch_size[1] == 0), \
               'Image dimensions must be divisible by the patch size.'

        num_patches = (self.image_size[0] // self.patch_size[0]) * \
                      (self.image_size[1] // self.patch_size[1])
        patch_dim = in_channels * self.patch_size[0] * self.patch_size[1]

        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, img):
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size[0], p2=self.patch_size[1])
        x = self.proj(x)
        return x

# --- Self-Attention Block ---
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---  Feed-Forward Network ---
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ---  Transformer Encoder Block ---
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)

    def forward(self, x):
        x = self.attn(x) + x # Residual connection
        x = self.ff(x) + x   # Residual connection
        return x

# ---  Causal Self-Attention Block (Modified for Causal LM)  ---
class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        h, n = self.heads, x.shape[1]

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        dots.masked_fill_(mask, float('-inf'))

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---  Transformer Decoder Block (for Causal LM, no cross-attention yet)  ---
class CausalTransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.attn = CausalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)

    def forward(self, x):
        x = self.attn(x) + x # Residual connection after causal self-attention
        x = self.ff(x) + x   # Residual connection after feed-forward
        return x

# ---  Cross-Attention Block (New) ---
class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, context):
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        kv = self.to_kv(context).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---  Multimodal Transformer Decoder Block (Modified with Cross-Attention) ---
class MultimodalTransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.causal_attn = CausalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.cross_attn = CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)

    def forward(self, x, visual_tokens):
        x = self.causal_attn(x) + x # Residual connection
        x = self.cross_attn(x, context=visual_tokens) + x # Residual connection
        x = self.ff(x) + x   # Residual connection
        return x
