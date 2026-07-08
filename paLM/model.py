import torch.nn as nn
from layers import ConceptualMultiModalInput,EncoderBlock

# 8. ConceptualPaLModel 
class ConceptualPaLModel(nn.Module):
    def __init__(
        self,
        text_vocab_size,
        text_embed_dim,
        image_feature_dim,
        common_embed_dim,
        max_seq_len,
        num_transformer_layers,
        num_heads, ff_dim,
        output_dim=None,
        dropout_rate=0.1
    ):
        super().__init__()
        # Reuse ConceptualMultilingualEmbedding for text input inside multimodal processor
        self.multimodal_input_processor = ConceptualMultiModalInput(
            text_vocab_size,
            text_embed_dim,
            image_feature_dim,
            common_embed_dim,
            max_seq_len
        )
        self.transformer_blocks = nn.ModuleList([
            EncoderBlock(common_embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_layers)
        ])
        self.output_head = nn.Linear(common_embed_dim, output_dim) if output_dim is not None else None

    def forward(self, text_tokens, image_features):
        combined_input_embeds = self.multimodal_input_processor(text_tokens, image_features)
        x = combined_input_embeds
        for block in self.transformer_blocks:
            x = block(x)
        if self.output_head is not None:
            return self.output_head(x)
        return x
