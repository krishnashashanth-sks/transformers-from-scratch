import torch.nn as nn
import torch
import torch.nn.functional as F

# --- Image Encoder Components ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        return x + self.pos_embed

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, self.mlp_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.mlp_hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(
            embed_dim,
            mlp_ratio=mlp_ratio,
            drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ImageEncoder(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = PositionalEmbedding(self.num_patches, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

# --- Prompt Encoder Components ---
class PointEncoder(nn.Module):
  def __init__(self, embed_dim:int, num_point_labels:int=2, scale:float=4.0):
    super().__init__()
    self.embed_dim=embed_dim
    self.num_point_labels=num_point_labels
    self.label_embedding=nn.Embedding(num_point_labels+1,embed_dim)
    self.coords_mlp=nn.Sequential(
        nn.Linear(2,embed_dim//2),
        nn.GELU(),
        nn.Linear(embed_dim//2,embed_dim)
    )
    self.scale=scale

  def forward(self, points:torch.Tensor, labels:torch.Tensor)->torch.Tensor:
    coords_embedded=self.coords_mlp(points/self.scale)
    labels_embedded=self.label_embedding(labels)
    return coords_embedded+labels_embedded

class BoxEncoder(nn.Module):
    def __init__(self, embed_dim:int, scale:float=4.0):
        super().__init__()
        self.embed_dim=embed_dim
        self.corner_mlp=nn.Sequential(
            nn.Linear(2,embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2,embed_dim)
        )
        self.scale = scale

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        top_left = boxes[:, :2]
        bottom_right = boxes[:, 2:]

        top_left_embedded = self.corner_mlp(top_left / self.scale)
        bottom_right_embedded = self.corner_mlp(bottom_right / self.scale)

        box_embedding = top_left_embedded + bottom_right_embedded

        return box_embedding

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_point_labels: int = 2, scale: float = 4.0,
                 num_transformer_blocks: int = 2, num_heads: int = 8, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.):
        super().__init__()
        self.embed_dim = embed_dim

        self.point_encoder = PointEncoder(embed_dim, num_point_labels, scale)
        self.box_encoder = BoxEncoder(embed_dim, scale)

        self.no_prompt_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.no_prompt_embed, std=.02)

        self.prompt_transformer = None
        if num_transformer_blocks > 0:
            self.prompt_transformer = nn.ModuleList([
                TransformerBlock(
                    embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
                )
                for _ in range(num_transformer_blocks)
            ])

    def forward(self, points: torch.Tensor = None, point_labels: torch.Tensor = None,
                boxes: torch.Tensor = None, points_mask: torch.Tensor = None, boxes_mask: torch.Tensor = None) -> torch.Tensor:

        all_embeddings = []
        batch_size = -1
        device = points.device if points is not None and points.numel() > 0 else (boxes.device if boxes is not None and boxes.numel() > 0 else self.no_prompt_embed.device)

        # Process point prompts
        if points is not None and point_labels is not None and points.numel() > 0:
            batch_size = points.shape[0]
            num_points_in_batch = points.shape[1]
            points_flat = points.view(batch_size * num_points_in_batch, 2)
            labels_flat = point_labels.view(batch_size * num_points_in_batch)
            point_embeds_flat = self.point_encoder(points_flat, labels_flat)
            point_embeds = point_embeds_flat.view(batch_size, num_points_in_batch, self.embed_dim)

            # Apply mask to zero out padded point embeddings
            if points_mask is not None:
                point_embeds = point_embeds * points_mask.unsqueeze(-1).to(point_embeds.dtype)
            all_embeddings.append(point_embeds)

        # Process box prompts
        if boxes is not None and boxes.numel() > 0:
            if batch_size == -1: # if not already set by points
                batch_size = boxes.shape[0]
            num_boxes_in_batch = boxes.shape[1] # Corrected variable name
            boxes_flat = boxes.view(batch_size * num_boxes_in_batch, 4) # Corrected variable name
            box_embeds_flat = self.box_encoder(boxes_flat)
            box_embeds = box_embeds_flat.view(batch_size, num_boxes_in_batch, self.embed_dim) # Corrected variable name

            # Apply mask to zero out padded box embeddings
            if boxes_mask is not None:
                box_embeds = box_embeds * boxes_mask.unsqueeze(-1).to(box_embeds.dtype)
            all_embeddings.append(box_embeds)

        if not all_embeddings:
            if batch_size == -1: # Default to batch size 1 if no prompts and no batch_size was inferred
                return self.no_prompt_embed.repeat(1, 1, 1).to(device)
            else:
                return self.no_prompt_embed.repeat(batch_size, 1, 1).to(device)

        # Concatenate all prompt embeddings
        prompt_embeddings = torch.cat(all_embeddings, dim=1)

        # Pass through Transformer blocks for interaction (optional)
        if self.prompt_transformer is not None:
            for blk in self.prompt_transformer:
                prompt_embeddings = blk(prompt_embeddings)

        return prompt_embeddings

# --- Mask Decoder Components ---
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        B, N_q, C_q = query.shape
        B, N_kv, C_kv = key_value.shape

        q = self.proj_q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.proj_kv(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C_q)

        x = self.proj_out(x)
        x = self.proj_drop(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn_image = CrossAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.cross_attn_prompt = CrossAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm4 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(embed_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, mask_queries, image_features, prompt_embeddings):
        q = mask_queries + self.self_attn(self.norm1(mask_queries))
        q = q + self.cross_attn_image(self.norm2(q), image_features)
        q = q + self.cross_attn_prompt(self.norm3(q), prompt_embeddings)
        q = q + self.mlp(self.norm4(q))
        return q

class MaskDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_mask_tokens: int = 4, num_decoder_layers: int = 2, num_heads: int = 8, mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0., output_upscale_factor: int = 4, image_feature_grid_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_mask_tokens = num_mask_tokens
        self.output_upscale_factor = output_upscale_factor
        self.image_feature_grid_size = image_feature_grid_size

        self.mask_token_embeddings = nn.Parameter(torch.randn(1, num_mask_tokens, embed_dim))
        nn.init.trunc_normal_(self.mask_token_embeddings, std=.02)

        self.mask_positional_embeddings = nn.Parameter(torch.randn(1, num_mask_tokens, embed_dim))
        nn.init.trunc_normal_(self.mask_positional_embeddings, std=.02)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(num_decoder_layers)
        ])

        self.final_cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, num_mask_tokens, kernel_size=1)
        )

        self.output_upsampler = nn.Upsample(scale_factor=output_upscale_factor // 4, mode='bilinear', align_corners=False)

    def forward(self, image_features: torch.Tensor, prompt_embeddings: torch.Tensor, original_image_size: tuple):
        batch_size = image_features.shape[0]
        mask_queries = self.mask_token_embeddings.repeat(batch_size, 1, 1) + \
                       self.mask_positional_embeddings.repeat(batch_size, 1, 1)

        for layer in self.decoder_layers:
            mask_queries = layer(mask_queries, image_features, prompt_embeddings)

        refined_mask_queries = mask_queries + self.final_cross_attn(self.final_norm(mask_queries), image_features)

        img_h_feat = self.image_feature_grid_size
        img_w_feat = self.image_feature_grid_size
        image_features_reshaped = image_features.transpose(1, 2).reshape(batch_size, self.embed_dim, img_h_feat, img_w_feat)

        low_res_masks = self.mask_head(image_features_reshaped)

        final_mask_logits = self.output_upsampler(low_res_masks)
        final_mask_logits = F.interpolate(final_mask_logits, size=original_image_size, mode='bilinear', align_corners=False)

        return final_mask_logits
