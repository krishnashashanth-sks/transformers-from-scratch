import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
  def __init__(self,image_size,patch_size,in_channels,embed_dim,num_heads,num_layers,mlp_dim):
    super().__init__()
    assert image_size % patch_size==0,'Image dimensions must be divisible by patchsize'
    self.patch_size=patch_size
    num_patches=(image_size//patch_size)**2
    self.positional_embedding=nn.Parameter(torch.randn(1,num_patches+1,embed_dim))
    self.patch_embedding=nn.Conv2d(
        in_channels,
        embed_dim,
        kernel_size=patch_size,
        stride=patch_size
    )
    encoder_layer=nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=mlp_dim,
        batch_first=True
    )
    self.transformer_encoder=nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers
    )
  def forward(self,x):
    x=self.patch_embedding(x) # Output shape: (B, embed_dim, H_out, W_out)
    x = x.flatten(2) # Flatten spatial dimensions: (B, embed_dim, H_out*W_out)
    x = x.transpose(1, 2) # Transpose to (B, H_out*W_out, embed_dim)

    # Add a class token if desired, and adjust positional embedding
    # For simplicity, here we just add positional embedding to patches
    num_patches = x.shape[1] # H_out*W_out
    # The positional embedding needs to be adapted for a class token if it were used.
    # For this simplified setup, we assume positional_embedding is already sized for patches.
    x = x + self.positional_embedding[:,:num_patches,:] # Adjust slicing if a class token is used

    return self.transformer_encoder(x)


class LanguageModel(nn.Module):
  def __init__(self,vocab_size,embed_dim,max_seq_len,num_heads,num_layers,mlp_dim):
    super().__init__()
    self.token_embedding=nn.Embedding(vocab_size,embed_dim)
    self.positional_embedding=nn.Parameter(torch.randn(1,max_seq_len,embed_dim))
    decoder_layer=nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=mlp_dim,
        batch_first=True
    )
    self.transformer_decoder=nn.TransformerDecoder(
        decoder_layer,
        num_layers=num_layers
    )
    self.fc_out=nn.Linear(embed_dim,vocab_size)
  def forward(self,text_tokens,visual_features):
    text_embeddings=self.token_embedding(text_tokens)
    seq_len=text_tokens.shape[1]
    text_embeddings=text_embeddings + self.positional_embedding[:,:seq_len,:]
    encoder_output=visual_features
    decoder_output=self.transformer_decoder(tgt=text_embeddings,memory=encoder_output)
    output_logits=self.fc_out(decoder_output)
    return output_logits


class ProjectionLayer(nn.Module):
  def __init__(self,input_dim,output_dim):
    super().__init__()
    self.projection=nn.Linear(input_dim,output_dim)
  def forward(self,x):
    return self.projection(x)
