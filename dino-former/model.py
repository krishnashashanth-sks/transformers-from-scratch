from layers import *
import torch.nn as nn
import torch

class VisionTransformer(nn.Module):
  def __init__(self,img_size=224,patch_size=16,in_chans=3,embed_dim=768, # Default in_chans to 3
               depth=12,num_heads=12,mlp_ratio=4.,qkv_bias=True,drop_rate=0.,
               attn_drop_rate=0.,norm_layer=nn.LayerNorm,**kwargs):
    super().__init__(**kwargs)
    self.embed_dim=embed_dim
    self.features=self.embed_dim
    self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
    num_patches=self.patch_embed.num_patches
    self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
    self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
    self.pos_drop=nn.Dropout(p=drop_rate)
    self.blocks=nn.ModuleList([
        Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer
        )
        for i in range(depth)
    ])
    self.norm=norm_layer(embed_dim)
    nn.init.trunc_normal_(self.pos_embed,std=.02)
    nn.init.trunc_normal_(self.cls_token,std=.02)
    self.apply(self._init_weight)
  def _init_weight(self,m):
    if isinstance(m,nn.Linear):
      nn.init.trunc_normal_(m.weight,std=.02)
      if isinstance(m,nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias,0)
    elif isinstance(m,nn.LayerNorm):
      nn.init.constant_(m.bias,0)
      nn.init.constant_(m.weight,1.0)
  def forward_features(self,x):
    B=x.shape[0]
    x=self.patch_embed(x)
    cls_token=self.cls_token.expand(B,-1,-1)
    x=torch.cat((cls_token,x),dim=1)
    x=self.pos_embed+x
    x=self.pos_drop(x)
    for blk in self.blocks:
      x=blk(x)
    x=self.norm(x)
    return x[:,0]
  def forward(self,x):
    return self.forward_features(x)

# ==============================================================================
# DINO Head
# ==============================================================================

class DINOHead(nn.Module):
  def __init__(self,in_dim,out_dim,hidden_dim_list=None,use_bn=True,norm_last_layer=True,n_last_blocks=1,patch_out_dim=0):
    super().__init__()
    hidden_dim_list=hidden_dim_list  if hidden_dim_list is not None else [2048]
    layers=[]
    last_dim=in_dim
    for hidden_dim in hidden_dim_list:
      layers.append(nn.Linear(last_dim,hidden_dim))
      if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim)) # Changed to BatchNorm1d
      layers.append(nn.GELU())
      last_dim=hidden_dim
    layers.append(nn.Linear(last_dim,out_dim))
    self.mlp=nn.Sequential(*layers)
    self.apply(self._init_weights)
  def _init_weights(self,m):
    if isinstance(m,nn.Linear):
      nn.init.trunc_normal_(m.weight,std=.02)
      if isinstance(m,nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias,0)
  def forward(self,x):
    x=self.mlp(x)
    return nn.functional.normalize(x,dim=-1,p=2)
