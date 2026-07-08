import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Vision Transformer (ViT) as Visual Encoder (Simplified) ---

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, num_heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
  def __init__(self,embed_dim,num_heads,mlp_ratio=4.,qkv_bias=False,drop=0.,attn_drop=0.,
               drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1=norm_layer(embed_dim)
    self.attn=Attention(embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop_path)
    self.norm2=norm_layer(embed_dim)
    mlp_hidden_dim=int(embed_dim*mlp_ratio)
    self.mlp=nn.Sequential(
        nn.Linear(embed_dim,mlp_hidden_dim),
        act_layer(),
        nn.Dropout(drop),
        nn.Linear(mlp_hidden_dim,embed_dim),
        nn.Dropout(drop)
    )
  def forward(self,x):
    x=x+self.attn(self.norm1(x))
    return x+self.mlp(self.norm2(x))
  
class VisualEncoder(nn.Module):
  def __init__(self,image_size=224,patch_size=16,in_channels=3,embed_dim=768,depth=12,num_heads=12):
    super().__init__()
    self.patch_embed=PatchEmbedding(in_channels,patch_size,embed_dim,image_size)
    self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
    self.pos_embed=nn.Parameter(torch.zeros(1,self.patch_embed.num_patches+1,embed_dim))
    self.pos_drop=nn.Dropout(p=0.1)
    self.blocks=nn.ModuleList(
        [Block(embed_dim,num_heads)for _ in range(depth)]
    )
    self.norm=nn.LayerNorm(embed_dim)
  def forward(self,x):
    B=x.shape[0]
    v=self.patch_embed(x)
    cls_tokens=self.cls_token.expand(B,-1,-1)
    x=torch.cat((cls_tokens,x),dim=1)
    x=x+self.pos_embed
    x=self.pos_drop(x)
    for blk in self.blocks:
      x=blk(x)
    x=self.norm(x)
    return x[:,0]
  
class LanguageEncoder(nn.Module):
  def __init__(self,vocab_size,embed_dim,num_heads,num_layers,max_len=77):
    super().__init__()
    self.token_embedding=nn.Embedding(vocab_size,embed_dim)
    self.position_embedding=nn.Embedding(max_len,embed_dim)
    self.transformer_blocks=nn.ModuleList([
        Block(embed_dim,num_heads) for _ in range(num_layers)
    ])
    self.norm=nn.LayerNorm(embed_dim)
  def forward(self,text_tokens):
    B,N=text_tokens.shape
    positions=torch.arange(N,device=text_tokens.device).unsqueeze(0)
    x=self.token_embedding(text_tokens)+self.position_embedding(positions)
    for blk in self.transformer_blocks:
      x=blk(x)
    return x
  
class CrossAttention(nn.Module):
  def __init__(self,query_dim,key_dim,value_dim,embed_dim,num_heads=8,qkv_bias=False,attn_drop=0.,proj_drop=0.):
    super().__init__()
    self.num_heads=num_heads
    head_dim=embed_dim//num_heads
    self.scale=head_dim**-0.5
    self.query=nn.Linear(query_dim,embed_dim,bias=qkv_bias)
    self.value=nn.Linear(value_dim,embed_dim,bias=qkv_bias)
    self.key=nn.Linear(key_dim,embed_dim,bias=qkv_bias)
    self.attn_drop=nn.Dropout(attn_drop)
    self.proj=nn.Linear(embed_dim,embed_dim)
    self.proj_drop=nn.Dropout(proj_drop)
  def forward(self,query_features,key_features,value_features):
    B_q,N_q,C_q=query_features.shape
    B_k,N_k,C_k=key_features.shape
    B_v,N_v,C_v=value_features.shape
    q=self.query(query_features).reshape(B_q,N_q,self.num_heads,C_q//self.num_heads).permute(0,2,1,3)
    k=self.key(key_features).reshape(B_k,N_k,self.num_heads,C_q//self.num_heads).permute(0,2,1,3)
    v=self.value(value_features).reshape(B_v,N_v,self.num_heads,C_v//self.num_heads).permute(0,2,1,3)
    attn=(q @ k.transpose(-2,-1))*self.scale
    attn=attn.softmax(dim=-1)
    attn=self.attn_drop(attn)
    x=(attn @ v).transpose(1,2).reshape(B_q,N_q,self.num_heads*(C_v//self.num_heads))
    return self.proj_drop(self.proj(x))

class MultimodalFusion(nn.Module):
  def __init__(self,visual_embed_dim,lang_embed_dim,num_fusion_layers=4,num_heads=8):
    super().__init__()
    self.cross_attn_blocks=nn.ModuleList([
        CrossAttention(query_dim=lang_embed_dim,key_dim=visual_embed_dim,value_dim=visual_embed_dim,embed_dim=lang_embed_dim,num_heads=num_heads)
        for _ in range(num_fusion_layers)
    ])
    self.mlp_blocks=nn.ModuleList([
        nn.Sequential(
            nn.LayerNorm(lang_embed_dim),
            nn.Linear(lang_embed_dim,lang_embed_dim*4),
            nn.GELU(),
            nn.Linear(lang_embed_dim*4,lang_embed_dim)
        )
        for _ in range(num_fusion_layers)
    ])
  def forward(self,visual_features,language_features):
    fused_features=language_features
    for i in range(len(self.cross_attn_blocks)):
      # Fix: key_features and value_features should both be visual_features
      fused_features=fused_features+self.cross_attn_blocks[i](
          query_features=fused_features,
          key_features=visual_features,
          value_features=visual_features
      )
      fused_features=fused_features+self.mlp_blocks[i](fused_features)
    return fused_features