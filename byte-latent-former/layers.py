import torch.nn as nn
import torch
import math
import torch.nn.functionl as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape (batch_size, seq_len, d_model)
        # self.pe has shape (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def _split_heads(self, x, batch_size):
        # x has shape (batch_size, seq_len, d_model)
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    @staticmethod
    def _scaled_dot_product_attention(q, k, v, mask=None, dropout_layer=None):
        # q, k, v have shape (batch_size, num_heads, seq_len, d_k)

        # Calculate attention scores (dot product of q and k's transpose)
        # (batch_size, num_heads, seq_len, d_k) @ (batch_size, num_heads, d_k, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Scale scores
        d_k = k.size(-1)
        scores = scores / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            # Ensure mask is broadcastable to scores (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        if dropout_layer is not None:
            attention_weights = dropout_layer(attention_weights)

        # Multiply attention weights with v
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_k) -> (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # 1. Apply linear transformations for Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 2. Split heads
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # 3. Perform scaled dot-product attention
        attn_output, _ = MultiHeadSelfAttention._scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 4. Concatenate heads
        # Transpose back to (batch_size, seq_len, num_heads, d_k) and then combine num_heads and d_k
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. Apply final linear output layer
        output = self.fc_out(attn_output)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x has shape (batch_size, seq_len, d_model)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, eps=1e-6):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer_output):
        # x is the input to the sublayer (e.g., attention or FFN)
        # sublayer_output is the output of the sublayer
        # Perform dropout on sublayer_output before adding
        return self.norm(x + self.dropout(sublayer_output))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, eps=1e-6):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout_rate)
        self.add_norm_1 = AddNorm(d_model, dropout_rate, eps)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        self.add_norm_2 = AddNorm(d_model, dropout_rate, eps)

    def forward(self, x, mask=None):
        # Multi-Head Self-Attention sub-layer
        attn_output = self.self_attn(x, mask)
        x = self.add_norm_1(x, attn_output) # Residual connection + LayerNorm

        # Feed-Forward Network sub-layer
        ffn_input = x # Store x before FFN for the second AddNorm
        ffn_output = self.ffn(x)
        x = self.add_norm_2(ffn_input, ffn_output) # Residual connection + LayerNorm

        return x

class EncoderTransformer(nn.Module):
  def __init__(self,vocab_size,d_model,num_heads,d_ff,num_layers,dropout_rate=0.1,max_len=5000,eps=1e-6):
    super(EncoderTransformer,self).__init__()
    self.d_model=d_model
    self.byte_embedding=nn.Embedding(vocab_size,d_model)
    self.pos_encoder=PositionalEncoding(d_model,max_len)
    self.dropout=nn.Dropout(dropout_rate)
    self.layers=nn.ModuleList([
        EncoderLayer(d_model,num_heads,d_ff,dropout_rate,eps)
        for _ in range(num_layers)
    ])
  def forward(self,x,mask=None):
    x=self.byte_embedding(x)*math.sqrt(self.d_model)
    x=self.pos_encoder(x)
    x=self.dropout(x)
    for layer in self.layers:
      x=layer(x,mask)
    return x

class LatentSpaceModule(nn.Module):
  def __init__(self,d_model,d_latent):
    super(LatentSpaceModule,self).__init__()
    self.mu_layer=nn.Linear(d_model,d_latent)
    self.log_var_layer=nn.Linear(d_model,d_latent)
  def forward(self,encoder_output):
    sequence_representation=encoder_output[:,0,:]
    mu=self.mu_layer(sequence_representation)
    log_var=self.log_var_layer(sequence_representation)
    std=torch.exp(0.5*log_var)
    eps=torch.randn_like(std) # Changed from torch.rand_like to torch.randn_like for standard normal sampling
    z=mu+eps*std
    return z,mu,log_var

class DecoderLayer(nn.Module):
  def __init__(self,d_model,num_heads,d_ff,dropout_rate=0.1,eps=1e-6):
    super(DecoderLayer,self).__init__()
    self.self_attn=MultiHeadSelfAttention(d_model,num_heads,dropout_rate)
    self.add_norm_1=AddNorm(d_model,dropout_rate,eps)
    self.cross_attn=MultiHeadSelfAttention(d_model,num_heads,dropout_rate)
    self.add_norm_2=AddNorm(d_model,dropout_rate,eps)
    self.ffn=FeedForwardNetwork(d_model,d_ff,dropout_rate)
    self.add_norm_3=AddNorm(d_model,dropout_rate,eps)

  def forward(self,x,enc_output,tgt_mask=None):
    attn_output=self.self_attn(x,tgt_mask)
    x=self.add_norm_1(x,attn_output)

    # Cross-attention
    # The enc_output here is actually the projected latent code for this VAE setup,
    # which comes from `model.latent_projection(latent_code).unsqueeze(1)`
    # The `_scaled_dot_product_attention` expects Q, K, V where K and V are from enc_output.
    # So, Query comes from decoder's current state (x), Key and Value come from enc_output.
    # Ensure batch_size is handled correctly when splitting heads for cross_attn
    batch_size_x = x.size(0)
    batch_size_enc = enc_output.size(0) # Should be the same as batch_size_x

    q_cross = self.cross_attn._split_heads(self.cross_attn.wq(x), batch_size_x)
    k_cross = self.cross_attn._split_heads(self.cross_attn.wk(enc_output), batch_size_enc)
    v_cross = self.cross_attn._split_heads(self.cross_attn.wv(enc_output), batch_size_enc)

    cross_attn_output, _ = MultiHeadSelfAttention._scaled_dot_product_attention(
        q_cross, k_cross, v_cross, mask=None, dropout_layer=self.cross_attn.dropout
    )

    # Concatenate heads and apply final linear layer for cross-attention output
    cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(batch_size_x, -1, self.cross_attn.d_model)
    cross_attn_output = self.cross_attn.fc_out(cross_attn_output)

    x=self.add_norm_2(x,cross_attn_output)

    ffn_output=self.ffn(x)
    return self.add_norm_3(x,ffn_output)


class DecoderTransformer(nn.Module):
  def __init__(self,vocab_size,d_model,num_heads,d_ff,num_layers,d_latent,dropout_rate=0.1,max_len=5000,eps=1e-6):
    super(DecoderTransformer,self).__init__()
    self.d_model=d_model
    self.byte_embedding=nn.Embedding(vocab_size,d_model)
    self.pos_encoder=PositionalEncoding(d_model,max_len)
    self.dropout=nn.Dropout(dropout_rate)
    self.latent_projection=nn.Linear(d_latent,d_model)
    self.layers=nn.ModuleList([
        DecoderLayer(d_model,num_heads,d_ff,dropout_rate,eps)
        for _ in range(num_layers)
    ])
  def forward(self,tgt,latent_code,tgt_mask=None):
    x=self.byte_embedding(tgt)*math.sqrt(self.d_model)
    x=self.pos_encoder(x)
    x=self.dropout(x)

    # Project latent code and unsqueeze to match the expected enc_output format for DecoderLayer
    # latent_code is (batch_size, d_latent)
    # projected_latent should be (batch_size, 1, d_model)
    projected_latent=self.latent_projection(latent_code).unsqueeze(1)

    for layer in self.layers:
      x=layer(x,projected_latent,tgt_mask)
    return x

class OutputHead(nn.Module):
  def __init__(self,d_model,vocab_size):
    super(OutputHead,self).__init__()
    self.linear=nn.Linear(d_model,vocab_size)
  def forward(self,x):
    return F.log_softmax(self.linear(x),dim=-1)
