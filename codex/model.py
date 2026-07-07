import torch.nn as nn
from layers import TokenEmbedding,PositionalEncoding,TransformerBlock

# Advanced Transformer Decoder Model
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        # self.embed_norm = nn.LayerNorm(embed_dim) # Removed this layer
        self.embed_dim = embed_dim # Still store embed_dim for other potential uses if needed

        self.decoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tokens, mask=None):
        # input_tokens shape: (batch_size, seq_len)
        # 1. Get token embeddings
        x = self.token_embedding(input_tokens)
        # 2. Add positional encodings
        x = self.positional_encoding(x)
        # 3. Apply dropout after combining embeddings and positional encoding
        # Removed embed_norm. The first LayerNorm will be inside TransformerBlock.
        x = self.dropout(x)

        # 4. Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, mask) # Pass mask to each decoder block

        # 5. Project to vocabulary size for next token prediction
        output = self.output_layer(x)
        return output