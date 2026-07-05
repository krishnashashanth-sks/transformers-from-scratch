import torch
import torch.nn as nn
import math
from functools import partial

# Helper for DropPath (Stochastic Depth)
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Stochastic depth, adapted from timm"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# Mlp (Feed-Forward Network)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Generic Multi-Head Self-Attention Module
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatiotemporalAttention(nn.Module):
    """
    Factorized Spatiotemporal Attention.
    Applies spatial attention, then temporal attention.
    """
    def __init__(
        self, dim, num_heads, num_temporal_patches, num_patches_per_frame,
        qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
    ):
        super().__init__()
        self.num_temporal_patches = num_temporal_patches
        self.num_patches_per_frame = num_patches_per_frame
        self.total_patches = num_temporal_patches * num_patches_per_frame

        self.spatial_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop
        )

        self.temporal_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop
        )

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.total_patches, f"Input token count {N} does not match expected total patches {self.total_patches}"

        x_spatial = x.view(B * self.num_temporal_patches, self.num_patches_per_frame, C)
        x_spatial = self.spatial_attn(x_spatial)
        x_spatial_reshaped = x_spatial.view(B, self.num_temporal_patches, self.num_patches_per_frame, C)

        x_temporal_input = x_spatial_reshaped.permute(0, 2, 1, 3).reshape(
            B * self.num_patches_per_frame, self.num_temporal_patches, C
        )
        x_temporal_output = self.temporal_attn(x_temporal_input)
        x_temporal_output_reshaped = x_temporal_output.view(
            B, self.num_patches_per_frame, self.num_temporal_patches, C
        ).permute(0, 2, 1, 3)

        output = x_temporal_output_reshaped.reshape(B, N, C)

        return output

class TimestepEmbedder(nn.Module):
    """Embeds timestep into a high-dimensional vector using sinusoidal embeddings."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = t.float() * math.pi
        freq_bands = torch.arange(0, self.frequency_embedding_size // 2, device=t.device).float() / (self.frequency_embedding_size // 2 - 1) * math.log(10000.0)
        freq_bands = torch.exp(freq_bands)

        emb = t_freq[:, None] * freq_bands[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        emb = self.mlp(emb)
        return emb

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization.
    Applies LayerNorm, then scales and shifts using conditioned parameters.
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, 2 * dim)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, cond):
        norm_x = self.norm(x)
        scale_shift = self.linear(cond)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        return norm_x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNTransformerBlock(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cond_dim=None,
        num_temporal_patches=None, num_patches_per_frame=None, use_spatiotemporal_attention=True
    ):
        super().__init__()
        assert cond_dim is not None, "cond_dim must be provided for AdaLNTransformerBlock"

        self.norm1 = AdaLayerNorm(dim, cond_dim)
        if use_spatiotemporal_attention:
            assert num_temporal_patches is not None and num_patches_per_frame is not None,\
                "num_temporal_patches and num_patches_per_frame must be provided for SpatiotemporalAttention"
            self.attn = SpatiotemporalAttention(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop,
                num_temporal_patches=num_temporal_patches,
                num_patches_per_frame=num_patches_per_frame
            )
        else:
            self.attn = Attention(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, t_embed):
        x = x + self.drop_path(self.attn(self.norm1(x, t_embed)))
        x = x + self.drop_path(self.mlp(self.norm2(x, t_embed)))
        return x

class SpatiotemporalPatchEmbed(nn.Module):
    """ Video to Patch Embedding with Tubelets and Positional Embeddings """
    def __init__(
        self, img_size=64, patch_size=8, in_channels=1, embed_dim=768, num_frames=10, tubelet_size=1, norm_layer=None
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        assert img_size[0] % patch_size[0] == 0, f"Image height {img_size[0]} not divisible by patch height {patch_size[0]}"
        assert img_size[1] % patch_size[1] == 0, f"Image width {img_size[1]} not divisible by patch width {patch_size[1]}"
        assert num_frames % tubelet_size == 0, f"Number of frames {num_frames} not divisible by tubelet size {tubelet_size}"

        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        num_temporal_patches = num_frames // tubelet_size

        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )

        self.num_spatial_patches_h = img_size[0] // patch_size[0]
        self.num_spatial_patches_w = img_size[1] // patch_size[1]
        self.num_temporal_patches = num_temporal_patches
        self.num_patches_per_frame = self.num_spatial_patches_h * self.num_spatial_patches_w
        self.total_patches = num_patches * num_temporal_patches

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.num_patches_per_frame, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.num_temporal_patches, embed_dim))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed_spatial, std=.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=.02)

    def forward(self, x):
        # x expected shape: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        B, C, T, H, W = x.shape

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # (B, total_patches, embed_dim)

        x_reshaped = x.view(B, self.num_temporal_patches, self.num_patches_per_frame, -1)
        x_reshaped = x_reshaped + self.pos_embed_spatial.unsqueeze(1)
        x_reshaped = x_reshaped + self.pos_embed_temporal.unsqueeze(2)
        x = x_reshaped.view(B, self.total_patches, -1)

        x = self.norm(x)
        return x

class PatchDecoder(nn.Module):
    """ Decodes token embeddings back into spatiotemporal patches. """
    def __init__(
        self, img_size=64, patch_size=8, in_channels=1, embed_dim=768, num_frames=10, tubelet_size=1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.num_temporal_patches = num_frames // tubelet_size
        self.num_spatial_patches_h = img_size // patch_size
        self.num_spatial_patches_w = img_size // patch_size
        self.num_patches_per_frame = self.num_spatial_patches_h * self.num_spatial_patches_w
        self.total_patches = self.num_patches_per_frame * self.num_temporal_patches

        self.proj_out = nn.Linear(embed_dim, tubelet_size * in_channels * patch_size * patch_size)

    def forward(self, x):
        # x expected shape: (B, total_patches, embed_dim)
        B, N, C_embed = x.shape
        assert N == self.total_patches, f"Input token count {N} does not match expected total patches {self.total_patches}"

        x = self.proj_out(x)

        x = x.view(B,
                   self.num_temporal_patches,
                   self.num_patches_per_frame,
                   self.tubelet_size * self.in_channels * self.patch_size * self.patch_size)

        x = x.view(B,
                   self.num_temporal_patches,
                   self.num_spatial_patches_h,
                   self.num_spatial_patches_w,
                   self.tubelet_size,
                   self.in_channels,
                   self.patch_size,
                   self.patch_size)

        x = x.permute(0, 1, 4, 5, 2, 6, 3, 7) # (B, T', T_tubelet, C, H', P_h, W', P_w)

        x = x.reshape(B,
                      self.num_frames,
                      self.in_channels,
                      self.num_spatial_patches_h * self.patch_size,
                      self.num_spatial_patches_w * self.patch_size)

        assert x.shape[-2] == self.img_size and x.shape[-1] == self.img_size,f"Reconstructed image size mismatch: expected ({self.img_size}, {self.img_size}), got ({x.shape[-2]}, {x.shape[-1]})"

        return x # (B, T, C, H, W)
