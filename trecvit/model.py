import torch
import torch.nn as nn
from layers import *

class TRecViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_frames,
                 num_gru_layers, num_attention_heads, mlp_dim, num_transformer_layers,
                 num_classes, dropout_rate=0.0, attention_dropout_rate=0.0):
        super().__init__()

        self.patch_embedding = VideoPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            dropout_rate=dropout_rate
        )

        # The input_dim to GRU should be embed_dim
        # The output of GRU will also be of hidden_dim size. If hidden_dim=embed_dim, then it matches.
        self.recurrent_module = RecurrentModule(
            input_dim=embed_dim,
            hidden_dim=embed_dim, # Output dimension of GRU is typically hidden_dim
            num_layers=num_gru_layers,
            dropout_rate=dropout_rate
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            mlp_dim=mlp_dim,
            num_layers=num_transformer_layers,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate
        )

        # Output Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (b, c, t, h, w)

        # 1. Video Patch Embedding
        # Output shape: (b, 1 (CLS) + t * num_spatial_patches, embed_dim)
        x = self.patch_embedding(x)

        # 2. Recurrent Module
        # Input shape: (b, seq_len, embed_dim)
        # Output shape: (b, seq_len, embed_dim) (if hidden_dim == embed_dim)
        x = self.recurrent_module(x)

        # 3. Transformer Encoder
        # Input/Output shape: (b, seq_len, embed_dim)
        x = self.transformer_encoder(x)

        # 4. Extract CLS token and pass through output head
        # The CLS token is the first element in the sequence dimension
        cls_token_output = x[:, 0] # shape: (b, embed_dim)

        # Apply LayerNorm and Linear head
        x = self.norm(cls_token_output)
        logits = self.head(x) # shape: (b, num_classes)

        return logits