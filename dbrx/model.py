import torch
import torch.nn as nn
from layers import EmbeddingLayer,TransformerBlock,NormalizationLayer

class DBRXModel(nn.Module):
    """The complete DBRX model architecture."""
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int,
                 num_transformer_blocks: int, num_heads: int, moe_hidden_dim: int,
                 num_experts: int, top_k: int, activation=nn.GELU, dropout: float = 0.1):
        super().__init__()

        self.embedding_layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                moe_hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
                dropout=dropout
            )
            for _ in range(num_transformer_blocks)
        ])

        self.final_norm = NormalizationLayer(embed_dim) # Final normalization layer
        self.output_layer = nn.Linear(embed_dim, vocab_size) # Maps embedding to vocabulary size

        self._init_weights() # Initialize weights

    def _init_weights(self):
        # Custom weight initialization for better training stability and performance
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding_layer(input_ids)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        x = self.final_norm(x)
        output = self.output_layer(x)

        return output
