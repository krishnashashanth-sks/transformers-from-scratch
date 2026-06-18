import torch
import torch.nn as nn
from einops import rearrange, repeat

class VideoPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_frames, dropout_rate=0.0):
        super().__init__()
        # Ensure image_size and patch_size are tuples
        if isinstance(image_size, int): # Make it flexible for single int or tuple
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, \
            "Image dimensions must be divisible by the patch size."

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.num_patches_h = image_size[0] // patch_size[0]
        self.num_patches_w = image_size[1] // patch_size[1]
        self.num_spatial_patches = self.num_patches_h * self.num_patches_w

        # Conv3d to extract patches and project to embed_dim
        # kernel_size=(1, patch_size[0], patch_size[1]) means we process each frame independently
        # stride=(1, patch_size[0], patch_size[1]) ensures non-overlapping spatial patches
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(1, patch_size[0], patch_size[1]),
            stride=(1, patch_size[0], patch_size[1])
        )

        # Spatial positional embedding: (num_patches_h * num_patches_w, embed_dim)
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, self.num_spatial_patches, embed_dim))

        # Temporal positional embedding: (num_frames, embed_dim)
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, num_frames, embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, in_channels, num_frames, image_height, image_width)
        b, c, t, h, w = x.shape

        # Apply Conv3d
        # Output shape: (b, embed_dim, t, num_patches_h, num_patches_w)
        patches = self.proj(x)

        # Rearrange to flatten spatial dimensions and combine with temporal
        # Resulting shape before adding positional embeddings: (b, t, num_spatial_patches, embed_dim)
        patches = rearrange(patches, 'b e t h w -> b t (h w) e')

        # Add spatial positional embedding
        # spatial_pos_embedding is (1, num_spatial_patches, embed_dim)
        # We need to expand it to (b, t, num_spatial_patches, embed_dim) for broadcasting
        patches = patches + self.spatial_pos_embedding.unsqueeze(1) # (b, t, num_spatial_patches, embed_dim)

        # Add temporal positional embedding
        # temporal_pos_embedding is (1, num_frames, embed_dim)
        # Expand it to (b, num_frames, 1, embed_dim) and then broadcast
        patches = patches + self.temporal_pos_embedding.unsqueeze(2) # (b, t, num_spatial_patches, embed_dim)

        # Flatten the temporal and spatial patch dimensions
        # Resulting shape: (b, t * num_spatial_patches, embed_dim)
        patches = rearrange(patches, 'b t n_s e -> b (t n_s) e')

        # Prepend CLS token
        # cls_token is (1, 1, embed_dim), repeat it for the batch dimension
        cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=b)
        x = torch.cat((cls_tokens, patches), dim=1)

        return self.dropout(x)

class RecurrentModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0 # Dropout is applied between layers
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        output, _ = self.gru(x)
        # output shape: (batch_size, sequence_length, hidden_dim)
        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.0, attention_dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout_rate, batch_first=False)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout_rate) # Dropout after final linear layer for FFN
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)

        # Multi-head self-attention block
        # LayerNorm before attention
        normalized_x = self.norm1(x)

        # MultiheadAttention expects (sequence_length, batch_size, embed_dim)
        # So we permute and then permute back
        normalized_x_permuted = normalized_x.permute(1, 0, 2) # (seq_len, batch_size, embed_dim)

        # attn_output shape: (sequence_length, batch_size, embed_dim)
        attn_output, _ = self.attn(query=normalized_x_permuted, key=normalized_x_permuted, value=normalized_x_permuted)

        # Permute back to (batch_size, sequence_length, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)

        # Residual connection + Dropout
        x = x + self.dropout1(attn_output)

        # Feed-forward network block
        # LayerNorm before FFN
        normalized_x = self.norm2(x)

        # Residual connection + FFN + Dropout
        x = x + self.mlp(normalized_x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers, dropout_rate=0.0, attention_dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate
            ))

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return x