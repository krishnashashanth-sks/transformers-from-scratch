import torch
import torch.nn as nn
from layers import PatchEmbedding,TransformerBlock,LayerNorm

# Main Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=1000, embed_dim=768,
                 depth=12, n_heads=12, mlp_ratio=4., dropout_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate
            ) for _ in range(depth)
        ])

        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()

        # Initialize positional embeddings
        # This is a simplified initialization. In practice, more sophisticated initializations are used.
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0] # Return the [CLS] token for classification/feature extraction

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

