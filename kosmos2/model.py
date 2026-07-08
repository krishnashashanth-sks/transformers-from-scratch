import torch.nn as nn
from layers import MultimodalTransformerDecoderBlock,PatchEmbedding,TransformerEncoderBlock
import torch
from utils import pair

# ---  Vision Transformer (Main Encoder) ---
class VisionTransformer(nn.Module):
    def __init__(self, *, image_size: int, patch_size: int, in_channels: int, num_classes: int = 1000,
                 embed_dim: int, depth: int, heads: int, mlp_dim: int, pool: str = 'cls',
                 dim_head: int = 64, dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (class token) or mean (mean pooling)'

        self.to_patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, image_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        ) if num_classes > 0 else nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img) # b, num_patches, embed_dim
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        if isinstance(self.mlp_head, nn.Identity):
            return x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
# --- Multimodal Text Transformer (Main Model with Visual Integration) (from Step 3) ---
class MultimodalTextTransformer(nn.Module):
    def __init__(self, *, vocab_size: int, max_seq_len: int, embed_dim: int, depth: int,
                 heads: int, mlp_dim: int, dim_head: int = 64, dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_blocks = nn.ModuleList([
            MultimodalTransformerDecoderBlock(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.to_logits = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, text_tokens, visual_tokens):
        b, n = text_tokens.shape

        x = self.token_emb(text_tokens) + self.pos_emb(torch.arange(n, device=text_tokens.device))
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, visual_tokens)

        x = self.norm(x)
        logits = self.to_logits(x)
        return logits