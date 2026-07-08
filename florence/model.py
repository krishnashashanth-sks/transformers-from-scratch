from layers import VisualEncoder,LanguageEncoder,MultimodalFusion
import torch
import torch.nn as nn

class FlorenceScratch(nn.Module):
  def __init__(self,image_size=224,patch_size=16,visual_embed_dim=768,visual_depth=12,visual_heads=12,
               vocab_size=30522,
               lang_embed_dim=768,lang_depth=12,lang_heads=12,
               fusion_layers=4,num_classes=1000,max_len=77):
    super().__init__()
    self.visual_encoder=VisualEncoder(image_size,patch_size,embed_dim=visual_embed_dim,depth=visual_depth,num_heads=visual_heads)
    self.language_encoder=LanguageEncoder(vocab_size,lang_embed_dim,lang_heads,lang_depth,max_len)
    self.fusion_module=MultimodalFusion(visual_embed_dim,lang_embed_dim,fusion_layers,lang_heads)
    self.vqa_head=nn.Sequential(
        nn.Linear(lang_embed_dim,lang_embed_dim*2),
                  nn.GELU(),
                  nn.Linear(lang_embed_dim*2,num_classes)
    )
  def forward(self,images,text_tokens):
    visual_all_tokens=self.visual_encoder.patch_embed(images)
    B=images.shape[0]
    cls_tokens=self.visual_encoder.cls_token.expand(B,-1,-1)
    visual_features=torch.cat((cls_tokens,visual_all_tokens),dim=1)
    visual_features=visual_features+self.visual_encoder.pos_embed
    for blk in self.visual_encoder.blocks:
      visual_features=blk(visual_features)
    visual_features=self.visual_encoder.norm(visual_features)
    language_features=self.language_encoder(text_tokens)
    fused_features=self.fusion_module(visual_features,language_features)
    pooled_fused_features=fused_features[:,0]
    return self.vqa_head(pooled_fused_features)