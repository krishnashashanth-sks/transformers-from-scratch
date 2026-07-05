import torch.nn as nn
import torch
from layers import ImageEncoder,PromptEncoder,MaskDecoder

# --- SAM2 Model Definition ---
class SAM2(nn.Module):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, image_encoder_depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0., num_point_labels: int = 2, prompt_encoder_scale: float = 4.0, num_prompt_transformer_blocks: int = 2, num_mask_tokens: int = 4, num_decoder_layers: int = 2, output_upscale_factor: int = 4, image_feature_grid_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.image_feature_grid_size = image_feature_grid_size

        self.image_encoder = ImageEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=image_encoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim, num_point_labels=num_point_labels, scale=prompt_encoder_scale, num_transformer_blocks=num_prompt_transformer_blocks, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate
        )

        self.mask_decoder = MaskDecoder(
            embed_dim=embed_dim, num_mask_tokens=num_mask_tokens, num_decoder_layers=num_decoder_layers, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, output_upscale_factor=output_upscale_factor, image_feature_grid_size=image_feature_grid_size
        )

    def forward(self, image: torch.Tensor, points: torch.Tensor = None, point_labels: torch.Tensor = None, boxes: torch.Tensor = None, original_image_size: tuple = None, points_mask: torch.Tensor = None, boxes_mask: torch.Tensor = None) -> torch.Tensor:

        if original_image_size is None:
            original_image_size = (self.img_size, self.img_size)

        image_features = self.image_encoder(image)

        prompt_embeddings = self.prompt_encoder(
            points=points, point_labels=point_labels, boxes=boxes, points_mask=points_mask, boxes_mask=boxes_mask
        )

        mask_logits = self.mask_decoder(
            image_features=image_features, prompt_embeddings=prompt_embeddings, original_image_size=original_image_size
        )

        return mask_logits

