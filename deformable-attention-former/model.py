import torch.nn as nn
from layers import PatchEmbed,DualAttentionBlock
import torch

class DualAttentionVisionTransformer(nn.Module):
  def __init__(self,img_size=224,patch_size=16,in_channels=3,
               num_classes=1000,embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.,
               qkv_bias=True,drop_rate=0.,attn_drop_rate=0.,
               group_size=7):
    super().__init__()
    self.num_classes=num_classes
    self.embed_dim=embed_dim
    self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim)
    num_patches=self.patch_embed.num_patches
    self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
    self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
    self.pos_drop=nn.Dropout(p=drop_rate)
    self.blocks=nn.ModuleList([DualAttentionBlock(
        dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
        drop=drop_rate,attn_drop=attn_drop_rate,group_size=group_size
    )
    for _ in range(depth)
    ])
    self.norm=nn.LayerNorm(embed_dim)
    self.head=nn.Linear(embed_dim,num_classes)
    nn.init.trunc_normal_(self.pos_embed,std=.02)
    nn.init.trunc_normal_(self.cls_token,std=.02)
    self.apply(self._init_weights)
  def _init_weights(self,m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
  def forward_features(self, x):
      B = x.shape[0]
      x = self.patch_embed(x)

      cls_token = self.cls_token.expand(B, -1, -1)  # Expand the class token to the batch size
      x = torch.cat((cls_token, x), dim=1)

      x = x + self.pos_embed
      x = self.pos_drop(x)

      for block in self.blocks:
          x = block(x)

      x = self.norm(x)
      return x[:, 0] # Return the class token output

  def forward(self, x):
      x = self.forward_features(x)
      x = self.head(x)
      return x