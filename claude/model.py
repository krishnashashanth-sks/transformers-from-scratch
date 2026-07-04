import torch
import torch.nn as nn
import math
from layers import TransformerDecoderBlock,RMSNorm

# Implementing ClaudeLikeModel
class ClaudeLikeModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_layers, num_heads, num_kv_heads, ffn_hidden_dim, dropout_rate):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)

        # Cache causal mask for efficiency (1 = allow, 0 = ignore)
        # Using 1 - triu creates a lower triangular matrix
        causal_mask_base = (1 - torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)).bool()
        self.register_buffer('causal_mask', causal_mask_base.view(1, 1, max_seq_len, max_seq_len))

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, L)
        batch_size, seq_len = input_ids.shape
        x = self.token_embeddings(input_ids) # (B, L, E)

        # 1. Slice the cached causal mask to match the current sequence length
        # Shape: (1, 1, L, L)
        current_causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # 2. Handle the attention_mask (padding mask)
        if attention_mask is not None:
            # attention_mask is (B, L) where 1 is valid, 0 is padding.
            # Reshape to (B, 1, 1, L) so it broadcasts across Heads and Queries
            padding_mask_expanded = attention_mask.view(batch_size, 1, 1, seq_len).bool()

            # Combine: Position is valid ONLY if it is both NOT future AND NOT padding
            final_mask = current_causal_mask & padding_mask_expanded
        else:
            final_mask = current_causal_mask

        # 3. Pass through layers
        for layer in self.layers:
            # We pass the final 4D mask down to GroupedQueryAttention
            x = layer(x, attention_mask=final_mask)

        x = self.norm(x)
        logits = self.output_projection(x)
        return logits