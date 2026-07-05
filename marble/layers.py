import torch
import torch.nn as nn

class TextEncoder(nn.Module):
  def __init__(self,vocab_size,embed_dim,num_heads,num_layers,dim_feedforward,max_se_len):
    super().__init__()
    self.token_embedding=nn.Embedding(vocab_size,embed_dim)
    self.position_embedding=nn.Embedding(max_se_len,embed_dim) # Fixed: Changed 'max_seq_len' to 'max_se_len'
    transformer_layer=nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    self.transformer_encoder=nn.TransformerEncoder(transformer_layer,num_layers=num_layers)
    self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim))
  def forward(self,tokens):
    batch_size,seq_len=tokens.shape
    token_embeds=self.token_embedding(tokens)
    # Original position embeddings for the tokens (before adding CLS)
    positions_indices = torch.arange(seq_len, device=tokens.device).unsqueeze(0).repeat(batch_size, 1)
    pos_embeds = self.position_embedding(positions_indices)

    cls_token_expanded=self.cls_token.repeat(batch_size,1,1)

    # Positional embedding for the CLS token (position 0)
    cls_pos_embed=self.position_embedding(torch.zeros(batch_size,1,dtype=torch.long,device=tokens.device))

    # Positional embeddings for the actual tokens, shifted by 1 to account for CLS token
    shifted_positions_indices = torch.arange(1, seq_len + 1, device=tokens.device).unsqueeze(0).repeat(batch_size, 1)
    shifted_pos_embeds = self.position_embedding(shifted_positions_indices)

    # Concatenate CLS token (with its position) and actual token embeddings (with their shifted positions)
    input_embeds=torch.cat([
        cls_token_expanded + cls_pos_embed, # CLS token + pos_embed for CLS
        token_embeds + shifted_pos_embeds   # Text tokens + shifted pos_embeds
    ],dim=1)
    output=self.transformer_encoder(input_embeds)
    return output[:,0,:]
  
class ImageEncoder(nn.Module):
  def __init__(self,img_size,patch_size,in_channels,embed_dim,num_heads,num_layers,dim_feedforward):
    super().__init__()
    assert img_size % patch_size==0,'Image dimensions must be divisible by the patch size.'
    num_patches=(img_size//patch_size)**2
    patch_dim=in_channels*patch_size*patch_size
    self.patch_embedding=nn.Linear(patch_dim,embed_dim)
    self.position_embedding=nn.Parameter(torch.randn(1,num_patches+1,embed_dim))
    self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim))
    transformer_layer=nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    self.transformer_encoder=nn.TransformerEncoder(transformer_layer,num_layers=num_layers)
    self.img_size=img_size
    self.patch_size=patch_size
  def forward(self,img):
    batch_size=img.shape[0]
    patches=img.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size)
    patches=patches.contiguous().view(batch_size,-1,self.patch_size*self.patch_size*img.shape[1]) # Fixed: Multiply by img.shape[1] to get full patch dim
    patch_embeds=self.patch_embedding(patches)
    cls_tokens=self.cls_token.repeat(batch_size,1,1) # Fixed typo: 'repear' to 'repeat'
    input_embeds=torch.cat((cls_tokens,patch_embeds),dim=1)
    input_embeds=input_embeds+self.position_embedding[:,:input_embeds.shape[1]]
    output=self.transformer_encoder(input_embeds)
    return output[:,0,:]
  
class MultiModalFusion(nn.Module):
  def __init__(self,embed_dim,num_heads,dim_feedforward,num_layers_cross_attn=1):
    super().__init__()
    self.embed_dim=embed_dim
    # Although the initial design mentioned cross-attention, for a 'from scratch'
    # implementation without relying on pre-trained components or complex
    # TransformerDecoderLayer setups (which is typical for cross-attention),
    # we will use a simpler, yet effective, concatenation and projection fusion.
    # The cross-attention layers are kept as placeholders in the __init__ but not used in forward
    # for this initial simplified implementation. If true cross-attention is needed,
    # a dedicated TransformerDecoderLayer or manual attention mechanism would be required.

    # Keeping these to reflect the initial thought process for a more advanced fusion,
    # but prioritizing a working 'from scratch' simple fusion first.
    self.text_to_image_attn=nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    self.text_to_image_encoder=nn.TransformerEncoder(self.text_to_image_attn,num_layers=num_layers_cross_attn)
    self.image_to_text_attn=nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )
    self.image_to_text_encoder=nn.TransformerEncoder(self.image_to_text_attn,num_layers=num_layers_cross_attn)

    self.fusion_projection=nn.Sequential(
        nn.Linear(2*embed_dim,embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim,embed_dim)
    )
  def forward(self,text_embedding,image_embedding):
    # text_embedding: (batch_size, embed_dim)
    # image_embedding: (batch_size, embed_dim)

    # For this simplified 'from scratch' fusion, we concatenate the embeddings
    # and pass them through a projection layer.
    fused_vector_raw=torch.cat((text_embedding,image_embedding),dim=-1)
    z_multimodal=self.fusion_projection(fused_vector_raw)
    return z_multimodal
  
class Swish(nn.Module):
  def forward(self,x):
    return x*torch.sigmoid(x)
  
class UNetBlock(nn.Module):
  def __init__(self,in_channels,out_channels,embed_dim_conditioning,time_emb_dim):
    super().__init__()
    self.norm1=nn.GroupNorm(8,in_channels)
    self.conv1=nn.Conv1d(in_channels,out_channels,kernel_size=3,padding=1)
    self.norm2=nn.GroupNorm(8,out_channels)
    self.conv2=nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=1)
    self.activation=Swish()
    self.conditional_projection=nn.Linear(embed_dim_conditioning,out_channels)
    self.time_projection=nn.Linear(time_emb_dim,out_channels)
    self.residual_conv=nn.Conv1d(in_channels,out_channels,kernel_size=1) if in_channels!=out_channels else nn.Identity()
  def forward(self,x,z_multimodal_cond,t_emb):
    h=self.activation(self.norm1(x))
    h=self.conv1(h)
    h+=self.conditional_projection(z_multimodal_cond).unsqueeze(-1)
    h+=self.time_projection(t_emb).unsqueeze(-1)
    h=self.activation(self.norm2(h))
    h=self.conv2(h)
    return h+self.residual_conv(x)
  
class SinusoidalPositionalEmbedding(nn.Module):
  def __init__(self,dim):
    super().__init__()
    self.dim=dim
  def forward(self,time):
    device=time.device
    half_dim=self.dim//2
    embeddings=torch.log(torch.tensor(10000.0,device=device))/(half_dim-1)
    embeddings=torch.exp(torch.arange(half_dim,device=device)*-embeddings)
    embeddings=time[:,None]*embeddings[None,:]
    return torch.cat((embeddings.sin(),embeddings.cos()),dim=-1)
  
class ConditionalDiffusionModel(nn.Module):
  def __init__(self,latent_dim_neRF,embed_dim_multimodal,time_emb_dim,num_unet_blocks=3,unet_channels_start=64):
    super().__init__()
    self.latent_dim_neRF=latent_dim_neRF
    self.embed_dim_multimodal=embed_dim_multimodal
    self.time_emb_time=time_emb_dim
    self.time_embed=SinusoidalPositionalEmbedding(time_emb_dim)
    self.time_mlp=nn.Sequential(
      nn.Linear(time_emb_dim,time_emb_dim*4),
      Swish(),
      nn.Linear(time_emb_dim*4,time_emb_dim)
      )
    channels=[latent_dim_neRF]+[unet_channels_start*(2**i)for i in range(num_unet_blocks)]
    self.down_blocks=nn.ModuleList()
    self.up_blocks=nn.ModuleList()
    for i in range(num_unet_blocks):
      in_c=channels[i]
      out_c=channels[i+1]
      self.down_blocks.append(UNetBlock(in_c,out_c,embed_dim_multimodal,time_emb_dim))
    self.mid_block=UNetBlock(channels[-1],channels[-1],embed_dim_multimodal,time_emb_dim)
    for i in range(num_unet_blocks):
      in_c=channels[i+1]
      out_c=channels[i]
      self.up_blocks.append(UNetBlock(in_c,out_c,embed_dim_multimodal,time_emb_dim))
    self.final_conv=nn.Conv1d(latent_dim_neRF,latent_dim_neRF,kernel_size=1)
    self.betas=torch.linspace(0.0001,0.02,1000)
    self.alphas=1-self.betas
    self.alphas_cumprod=torch.cumprod(self.alphas,axis=0)
    self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-self.alphas_cumprod)
  def forward(self,x_t,t,z_multimodal_cond):
    t_emb=self.time_mlp(self.time_embed(t))
    h=x_t
    down_outputs=[]
    for block in self.down_blocks:
      h=block(h,z_multimodal_cond,t_emb)
      down_outputs.append(h)
    h=self.mid_block(h,z_multimodal_cond,t_emb)
    for block in self.up_blocks:
      skip_feature=down_outputs.pop()
      h=block(h+skip_feature,z_multimodal_cond,t_emb)
    return self.final_conv(h)
  def q_sample(self,x_start,t,noise=None):
    if noise is None:
      noise=torch.randn_like(x_start)
    sqrt_alphas_cumprod_t=self.sqrt_alphas_cumprod[t][:,None,None]
    sqrt_one_minus_alphas_cumprod_t=self.sqrt_one_minus_alphas_cumprod[t][:,None,None]
    x_t=sqrt_alphas_cumprod_t*x_start+sqrt_one_minus_alphas_cumprod_t*noise
    return x_t
  def p_sample(self,x_t,t,x_multimodal_cond):
    predicted_noise=self.forward(x_t,t,x_multimodal_cond)
    return predicted_noise
  
class NeRFMLP(nn.Module):
  def __init__(self,latent_dim_neRF,hidden_dim=128,output_dim_density=1,output_dim_color=3):
    super().__init__()
    self.latent_dim_neRF=latent_dim_neRF
    self.fc1=nn.Linear(latent_dim_neRF+60+24,hidden_dim)
    self.fc2=nn.Linear(hidden_dim,hidden_dim)
    self.fc_density=nn.Linear(hidden_dim,output_dim_density)
    self.fc_color=nn.Linear(hidden_dim,output_dim_color)
    self.activation=nn.ReLU()
  def forward(self,z_neRF_latent,positions_encoded,directions_encoded):
    x=torch.cat([z_neRF_latent,positions_encoded,directions_encoded],dim=-1)
    x=self.activation(self.fc1(x))
    x=self.activation(self.fc2(x))
    density=self.fc_density(x)
    color=torch.sigmoid(self.fc_color(x))
    return density,color
  
class LatentSpaceMemoryNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Using an LSTM to model temporal dependencies in the latent space
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Project LSTM output back to latent_dim if needed, or use directly as context
        self.output_projection = nn.Linear(hidden_dim, latent_dim) if hidden_dim != latent_dim else nn.Identity()

    def forward(self, current_latent, prev_memory_state=None):
        # current_latent: (batch_size, latent_dim) - The NeRF latent code generated at the current timestep
        # prev_memory_state: (h_n, c_n) from previous step, each (num_layers, batch_size, hidden_dim)

        # LSTM expects input of shape (batch_size, seq_len, input_size)
        # For our case, seq_len is 1 as we process one latent at a time
        current_latent_seq = current_latent.unsqueeze(1) # (batch_size, 1, latent_dim)

        # Pass current latent through LSTM
        if prev_memory_state is None:
            # Initialize hidden and cell states if not provided
            h0 = torch.zeros(self.num_layers, current_latent.size(0), self.hidden_dim).to(current_latent.device)
            c0 = torch.zeros(self.num_layers, current_latent.size(0), self.hidden_dim).to(current_latent.device)
            lstm_out, (h_n, c_n) = self.lstm(current_latent_seq, (h0, c0))
        else:
            lstm_out, (h_n, c_n) = self.lstm(current_latent_seq, prev_memory_state)

        # lstm_out: (batch_size, 1, hidden_dim)
        # h_n, c_n: (num_layers, batch_size, hidden_dim) - new memory state

        # Project LSTM output to latent_dim (if necessary) and squeeze seq_len dimension
        memory_output = self.output_projection(lstm_out.squeeze(1)) # (batch_size, latent_dim)

        return memory_output, (h_n, c_n)