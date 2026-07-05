import torch.nn as nn
from layers import SpatiotemporalPatchEmbed,TimestepEmbedder,AdaLNTransformerBlock,PatchDecoder
import torch
from functools import partial

class STDiT(nn.Module):
    def __init__(
        self, img_size=64, patch_size=8, in_channels=1, embed_dim=768, num_frames=10, tubelet_size=2,
        depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1,
        attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.num_heads = num_heads
        self.depth = depth

        # 1. Spatiotemporal Patch Embedding
        self.patch_embed = SpatiotemporalPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size, norm_layer=norm_layer
        )
        self.total_patches = self.patch_embed.total_patches
        self.num_temporal_patches = self.patch_embed.num_temporal_patches
        self.num_patches_per_frame = self.patch_embed.num_patches_per_frame

        # 2. Time Step Embedding
        self.t_embedder = TimestepEmbedder(hidden_size=embed_dim)

        # 3. AdaLN Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                cond_dim=embed_dim, # Conditioning dimension from time embedding
                num_temporal_patches=self.num_temporal_patches,
                num_patches_per_frame=self.num_patches_per_frame,
                use_spatiotemporal_attention=True
            )
            for i in range(depth)
        ])

        # 4. Final Layer Norm
        self.final_layer_norm = norm_layer(embed_dim)

        # 5. Patch Decoder
        self.patch_decoder = PatchDecoder(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, num_frames=num_frames, tubelet_size=tubelet_size
        )

        # Initialize weights for final projection if needed, or rely on default from nn.Linear

    def forward(self, x, t):
        # x: (B, T, C, H, W) - e.g., noisy video frames
        # t: (B,) - diffusion time steps

        # 1. Embed spatiotemporal input
        x = self.patch_embed(x) # (B, total_patches, embed_dim)

        # 2. Embed time steps
        t_embed = self.t_embedder(t) # (B, embed_dim)

        # 3. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x, t_embed) # (B, total_patches, embed_dim)

        # 4. Final Layer Norm
        x = self.final_layer_norm(x)

        # 5. Patch Decoding
        output = self.patch_decoder(x) # (B, T, C, H, W)

        return output