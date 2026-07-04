import torch.nn as nn
import torch
from layers import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embeddings (learned positional embeddings)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        # Dropout for embeddings
        self.dropout_emb = nn.Dropout(dropout_rate)

        # Stack of Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_dim)

        # Language model head (linear layer to project to vocab size)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Share weights between token embedding and language model head
        # (Common practice in many transformer models like GPT-2, GPT-3)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Positional embeddings
        # Create positional IDs from 0 to seq_len-1 for each example in the batch
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)

        # Combine token and positional embeddings
        x = self.dropout_emb(token_embeds + position_embeds)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)

        # Apply final layer normalization
        x = self.final_norm(x)

        # Language model head for next token prediction
        logits = self.lm_head(x)
        return logits