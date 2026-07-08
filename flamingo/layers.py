import torch.nn as nn
import torch

class PreNorm(nn.Module):
  def __init__(self,dim,fn):
    super().__init__()
    self.norm=nn.LayerNorm(dim)
    self.fn=fn
  def forward(self,x,*args,**kwargs):
    return self.fn(self.norm(x),*args,**kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
  def __init__(self,dim,heads=8,dim_head=64,dropout=0.,is_cross_attention=False):
    super().__init__()
    inner_dim=dim_head*heads
    project_out=not(heads==1 and dim_head==dim)
    self.heads=heads
    self.scale=dim_head**-0.5
    self.is_cross_attention = is_cross_attention
    self.inner_dim = inner_dim

    self.to_q=nn.Linear(dim,inner_dim,bias=False)
    if is_cross_attention:
      self.to_kv=nn.Linear(dim,inner_dim*2,bias=False)
    else:
      self.to_kv=nn.Linear(dim,inner_dim*2,bias=False)

    self.to_out=nn.Sequential(
        nn.Linear(inner_dim,dim),
        nn.Dropout(dropout)
    )if project_out else nn.Identity()

  def forward(self,x,context=None):
    h=self.heads
    q=self.to_q(x)

    if self.is_cross_attention:
      if context is None:
        raise ValueError("Context must be provided for cross-attention.")
      kv_input = context
    else:
      kv_input = x

    kv=self.to_kv(kv_input).chunk(2,dim=-1)
    k,v=map(lambda t:t.view(t.shape[0],-1,h,t.shape[-1]//h).transpose(1,2),kv)
    q=q.view(q.shape[0],-1,h,q.shape[-1]//h).transpose(1,2)

    dots=torch.matmul(q,k.transpose(-1,-2))*self.scale
    attn=dots.softmax(dim=-1)
    out=torch.matmul(attn,v)

    out=out.transpose(1,2).contiguous().view(out.shape[0],-1,self.inner_dim)
    return self.to_out(out)

# --- Vision Encoder Components --- #

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.,
        ff_hidden_mult: int = 4
    ):
        super().__init__()
        self.self_attn = PreNorm(
            dim,
            Attention(
                dim,
                heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                is_cross_attention=False
            )
        )
        self.feed_forward = PreNorm(
            dim,
            FeedForward(
                dim,
                dim * ff_hidden_mult,
                dropout=dropout
            )
        )

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.feed_forward(x)
        return x

class VisionEncoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.,
        ff_hidden_mult: int = 4,
        has_cls_token: bool = True
    ):
        super().__init__()
        self.has_cls_token = has_cls_token

        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embedding.num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + (1 if has_cls_token else 0), embed_dim)
        )

        if has_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer_encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                ff_hidden_mult=ff_hidden_mult
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embedding(x)

        if self.has_cls_token:
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding

        for block in self.transformer_encoder_blocks:
            x = block(x)

        x = self.norm(x)

        if self.has_cls_token:
            return x[:, 0]
        else:
            return x

# --- Perceiver Resampler --- #

class PerceiverSampler(nn.Module):
  def __init__(
      self,
      num_latents:int,
      latent_dim:int,
      input_dim:int,
      num_cross_attention_heads:int=8,
      num_self_attention_heads:int=8,
      num_cross_attention_layers:int=1,
      num_self_attention_layers:int=6,
      cross_attention_dropout:float=0.,
      self_attention_dropout:float=0.,
      ff_dropout:float=0.,
      ff_hidden_mult:int=4,
      dim_head:int=64
  ):
    super().__init__()
    self.num_latents=num_latents
    self.latent_dim=latent_dim
    self.input_dim=input_dim
    self.latents=nn.Parameter(torch.randn(num_latents,latent_dim))
    self.cross_attn_blocks=nn.ModuleList([
        nn.Sequential(
            PreNorm(latent_dim,Attention(latent_dim,heads=num_cross_attention_heads,dim_head=dim_head,dropout=cross_attention_dropout,is_cross_attention=True)),
            PreNorm(latent_dim,FeedForward(latent_dim,latent_dim*ff_hidden_mult,dropout=ff_dropout))
        )
        for _ in range(num_cross_attention_layers)
    ])
    self.self_attn_blocks=nn.ModuleList([
        nn.Sequential(
            PreNorm(latent_dim,Attention(latent_dim,heads=num_self_attention_heads,dim_head=dim_head,dropout=self_attention_dropout)),
            PreNorm(latent_dim,FeedForward(latent_dim,latent_dim*ff_hidden_mult,dropout=ff_dropout))
        )
        for _ in range(num_self_attention_layers)
    ])
    self.input_projection=nn.Linear(input_dim,latent_dim) if input_dim!=latent_dim else nn.Identity()
  def forward(self,visual_features):
    batch_size=visual_features.shape[0]
    visual_features=self.input_projection(visual_features)
    latents=self.latents.unsqueeze(0).repeat(batch_size,1,1)
    for cross_attn_block in self.cross_attn_blocks:
      cross_attn_output=cross_attn_block[0](latents,context=visual_features)
      latents=latents+cross_attn_output
      latents=latents+cross_attn_block[1](latents)
    for self_attn_block in self.self_attn_blocks:
      self_attn_output=self_attn_block[0](latents)
      latents=latents+self_attn_output
      latents=latents+self_attn_block[1](latents)
    return latents

# --- Language Model Components --- #

class LanguageModelDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        dropout: float,
        ff_hidden_mult: int
    ):
        super().__init__()
        self.self_attn = PreNorm(
            dim,
            Attention(
                dim,
                heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                is_cross_attention=False
            )
        )
        self.self_ff = PreNorm(
            dim,
            FeedForward(
                dim,
                dim * ff_hidden_mult,
                dropout=dropout
            )
        )

        self.cross_attn = PreNorm(
            dim,
            Attention(
                dim,
                heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                is_cross_attention=True
            )
        )
        self.cross_ff = PreNorm(
            dim,
            FeedForward(
                dim,
                dim * ff_hidden_mult,
                dropout=dropout
            )
        )

    def forward(self, x, visual_tokens):
        self_attn_out = self.self_attn(x)
        x = x + self_attn_out
        x = x + self.self_ff(x)

        cross_attn_out = self.cross_attn(x, context=visual_tokens)
        x = x + cross_attn_out
        x = x + self.cross_ff(x)

        return x

class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        dim: int,
        num_decoder_blocks: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.,
        ff_hidden_mult: int = 4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Embedding(max_seq_len, dim)

        self.decoder_blocks = nn.ModuleList([
            LanguageModelDecoderBlock(
                dim=dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                ff_hidden_mult=ff_hidden_mult
            )
            for _ in range(num_decoder_blocks)
        ])

        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, text_tokens, visual_tokens):
        x = self.token_embedding(text_tokens)

        seq_len = text_tokens.shape[1]
        position_ids = torch.arange(0, seq_len, device=text_tokens.device).unsqueeze(0)
        position_embeddings = self.positional_embedding(position_ids)

        lm_hidden_states = x + position_embeddings

        for block in self.decoder_blocks:
            lm_hidden_states = block(lm_hidden_states, visual_tokens)

        logits = self.to_logits(lm_hidden_states)

        return logits
