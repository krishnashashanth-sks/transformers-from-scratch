import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # (B, H*W, C)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LocalAttention(nn.Module):
  def __init__(self,dim,num_heads=8,group_size=7,qkv_bias=False,attn_drop=0.,proj_drop=0.):
    super().__init__()
    self.num_heads=num_heads
    self.group_size=group_size
    self.head_dim=dim//num_heads
    self.scale=self.head_dim**-0.5
    self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
    self.attn_drop=nn.Dropout(attn_drop)
    self.proj=nn.Linear(dim,dim)
    self.proj_drop=nn.Dropout(proj_drop)
  def forward(self,x):
    B,N,C=x.shape
    H=W=int(N**0.5)
    if H % self.group_size != 0:
      raise ValueError(f"Patch dimension {H} must be divisible by group_size {self.group_size}")

    # Reshape for grouped local attention
    x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
    x_grouped = rearrange(x, 'b (h_g gs1) (w_g gs2) c -> (b h_g w_g) (gs1 gs2) c',
                          gs1=self.group_size, gs2=self.group_size) # (BG, GS, C)

    BG, GS, _ = x_grouped.shape # Get grouped batch size and group sequence length

    # QKV projection and split
    qkv = self.qkv(x_grouped).reshape(BG, GS, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2] # (BG, num_heads, GS, head_dim)

    # Attention calculation
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Concatenate heads and project
    x_attn_out = (attn @ v).transpose(1, 2).reshape(BG, GS, C) # (BG, GS, C)
    x_attn_out = self.proj(x_attn_out)
    x_attn_out = self.proj_drop(x_attn_out)

    # Reshape back to original sequence format
    # Undo the rearrangement: (b h w c) -> (b (h w) c)
    x_ungrouped = rearrange(x_attn_out, '(b h_g w_g) (gs1 gs2) c -> b (h_g gs1) (w_g gs2) c',
                            b=B, h_g=H//self.group_size, w_g=W//self.group_size,
                            gs1=self.group_size, gs2=self.group_size)
    x_final = rearrange(x_ungrouped, 'b h w c -> b (h w) c')
    return x_final
  
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )
    def forward(self, x):
        return self.net(x)
    
class DualAttentionBlock(nn.Module):
  def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=False,attn_drop=0.,drop=0.,group_size=7):
    super().__init__()
    self.norm1=nn.LayerNorm(dim)
    self.global_attn=MultiHeadSelfAttention(dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)
    self.local_attn=LocalAttention(dim,num_heads=num_heads,group_size=group_size,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)
    self.norm2=nn.LayerNorm(dim)
    mlp_hidden_dim=int(dim*mlp_ratio)
    self.mlp=FeedForward(dim,mlp_hidden_dim,drop=drop)
    self.fusion_method='add'
  def forward(self,x):
    identity=x
    x_norm=self.norm1(x)
    patch_tokens_for_local=x_norm[:,1:]
    output_global=self.global_attn(x_norm)
    output_local_patches=self.local_attn(patch_tokens_for_local)
    fused_patches=output_global[:,1:]+output_local_patches
    fused_sequence=torch.cat((output_global[:,0:1],fused_patches),dim=1)
    if self.fusion_method=='add':
      x=identity+fused_sequence
    elif self.fusion_method=='concat':
      raise NotImplementedError("Concatenataion fusion requires a projection layers adter concatenation")
    else:
      raise ValueError("Unsupported fusion method")
    x=x+self.mlp(self.norm2(x))
    return x