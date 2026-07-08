import torch.nn as nn
from layers import VisionEncoder,PerceiverSampler,LanguageModel

# --- Full Flamingo-Like Model --- #

class FlamingoLikeModel(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        vision_embed_dim: int,
        vision_depth: int,
        vision_num_heads: int,
        vision_dim_head: int,
        vision_dropout: float,
        vision_ff_hidden_mult: int,
        perceiver_num_latents: int,
        perceiver_latent_dim: int,
        perceiver_num_cross_attention_heads: int,
        perceiver_num_self_attention_heads: int,
        perceiver_num_cross_attention_layers: int,
        perceiver_num_self_attention_layers: int,
        perceiver_cross_attention_dropout: float,
        perceiver_self_attention_dropout: float,
        perceiver_ff_dropout: float,
        perceiver_ff_hidden_mult: int,
        perceiver_dim_head: int,
        vocab_size: int,
        max_seq_len: int,
        language_dim: int,
        language_num_decoder_blocks: int,
        language_num_heads: int,
        language_dim_head: int,
        language_dropout: float,
        language_ff_hidden_mult: int
    ):
        super().__init__()

        assert vision_embed_dim == perceiver_latent_dim == language_dim,"Embedding dimensions of VisionEncoder, PerceiverSampler, and LanguageModel must match."

        self.vision_encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            dim_head=vision_dim_head,
            dropout=vision_dropout,
            ff_hidden_mult=vision_ff_hidden_mult,
            has_cls_token=False
        )

        self.perceiver_sampler = PerceiverSampler(
            num_latents=perceiver_num_latents,
            latent_dim=perceiver_latent_dim,
            input_dim=vision_embed_dim,
            num_cross_attention_heads=perceiver_num_cross_attention_heads,
            num_self_attention_heads=perceiver_num_self_attention_heads,
            num_cross_attention_layers=perceiver_num_cross_attention_layers,
            num_self_attention_layers=perceiver_num_self_attention_layers,
            cross_attention_dropout=perceiver_cross_attention_dropout,
            self_attention_dropout=perceiver_self_attention_dropout,
            ff_dropout=perceiver_ff_dropout,
            ff_hidden_mult=perceiver_ff_hidden_mult,
            dim_head=perceiver_dim_head
        )

        self.language_model = LanguageModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dim=language_dim,
            num_decoder_blocks=language_num_decoder_blocks,
            num_heads=language_num_heads,
            dim_head=language_dim_head,
            dropout=language_dropout,
            ff_hidden_mult=language_ff_hidden_mult
        )

    def forward(self, pixel_values, lang_tokens):
        visual_features = self.vision_encoder(pixel_values)
        visual_tokens = self.perceiver_sampler(visual_features)
        logits = self.language_model(lang_tokens, visual_tokens)
        return logits