import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Injects positional encoding into input embeddings."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Adds positional encoding to the input embeddings x.

        Args:
            x (Tensor): Input embeddings of shape (seq_len, batch_size, d_model).

        Returns:
            Tensor: Embeddings with positional encoding added.
        """
        # Ensure positional encoding is broadcastable
        x = x + self.pe[:x.size(0), :]
        return x

def quantize_tensor(x, num_bits=8, scale=None, zero_point=None):
    if scale is None or zero_point is None:
        # Simple min-max quantization for demonstration
        # Detach x.min() and x.max() from the graph and convert to scalar values
        x_min = x.min().detach().item()
        x_max = x.max().detach().item()

        scale = (x_max - x_min) / (2**num_bits - 1)
        zero_point = x_min

    # Clip values to quantization range
    x_quant = torch.round((x - zero_point) / scale)
    x_quant = torch.clamp(x_quant, 0, 2**num_bits - 1)

    # Dequantize (simulate quantized behavior)
    x_dequant = x_quant * scale + zero_point
    return x_dequant, scale, zero_point

class QuantizedMultiHeadAttention(nn.Module):
    """Quantized Multi-Head Attention mechanism."""
    def __init__(self, d_model, num_heads, dropout=0.1, num_bits=8):
        super(QuantizedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.num_bits = num_bits # Number of bits for quantization

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Initialize quantization parameters for each linear layer output if needed
        self.q_scale, self.q_zero_point = None, None
        self.k_scale, self.k_zero_point = None, None
        self.v_scale, self.v_zero_point = None, None
        self.attn_scores_scale, self.attn_scores_zero_point = None, None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Project and split into heads, then apply quantization
        # Query
        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        q, self.q_scale, self.q_zero_point = quantize_tensor(q, self.num_bits, self.q_scale, self.q_zero_point)

        # Key
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k, self.k_scale, self.k_zero_point = quantize_tensor(k, self.num_bits, self.k_scale, self.k_zero_point)

        # Value
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v, self.v_scale, self.v_zero_point = quantize_tensor(v, self.num_bits, self.v_scale, self.v_zero_point)

        # 2) Apply scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Quantize attention scores
        scores, self.attn_scores_scale, self.attn_scores_zero_point = quantize_tensor(scores, self.num_bits, self.attn_scores_scale, self.attn_scores_zero_point)

        if mask is not None:
            # Replace in-place masked_fill_ with out-of-place torch.where
            scores = torch.where(mask == 0, torch.full_like(scores, -1e9), scores) # Fill with a very small number for softmax

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, v)

        # 3) Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Output projection is not quantized here as per general Transformer practice, but can be if required.
        return self.linear_out(x)

class QuantizedFeedForward(nn.Module):
    """A quantized position-wise feed-forward network."""
    def __init__(self, d_model, d_ff, dropout=0.1, num_bits=8):
        super(QuantizedFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.num_bits = num_bits

        # Quantization parameters for intermediate activations
        self.activation_scale, self.activation_zero_point = None, None

    def forward(self, x):
        # First linear layer
        x = self.w_1(x)

        # Apply activation function (e.g., ReLU) and then quantize
        x = F.relu(x)
        x, self.activation_scale, self.activation_zero_point = quantize_tensor(x, self.num_bits, self.activation_scale, self.activation_zero_point)

        # Apply dropout
        x = self.dropout(x)

        # Second linear layer
        return self.w_2(x)

class SublayerConnection(nn.Module):
  def __init__(self,size,dropout):
    super(SublayerConnection,self).__init__()
    self.norm=nn.LayerNorm(size)
    self.dropout=nn.Dropout(dropout)
  def forward(self,x,sublayer):
    return x+self.dropout(sublayer(self.norm(x)))
  
class EncoderLayer(nn.Module):
  def __init__(self,size,self_attn,feed_forward,dropout):
    super(EncoderLayer,self).__init__()
    self.self_attn=self_attn
    self.feed_forward=feed_forward
    self.sublayer=nn.ModuleList([SublayerConnection(size,dropout) for _ in range(2)])
    self.size=size
  def forward(self,x,mask):
    x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
    return self.sublayer[1](x,self.feed_forward)

class Encoder(nn.Module):
  def __init__(self,layer,N):
    super(Encoder,self).__init__()
    self.layers=nn.ModuleList([layer for _ in range(N)])
    self.norm=nn.LayerNorm(layer.size)
  def forward(self,x,mask):
    for layer in self.layers:
      x=layer(x,mask)
    return self.norm(x)
  
class DecoderLayer(nn.Module):
  def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
    super(DecoderLayer,self).__init__()
    self.size=size
    self.self_attn=self_attn
    self.src_attn=src_attn
    self.feed_forward=feed_forward
    self.sublayer=nn.ModuleList([SublayerConnection(size,dropout) for _ in range(3)])
  def forward(self,x,memory,src_mask,tgt_mask):
    m=memory
    x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
    x=self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
    return self.sublayer[2](x,self.feed_forward)
  
class Decoder(nn.Module):
  def __init__(self,layer,N):
    super(Decoder,self).__init__()
    self.layers=nn.ModuleList([layer for _ in range(N)])
    self.norm=nn.LayerNorm(layer.size)
  def forward(self,x,memory,src_mask,tgt_mask):
    for layer in self.layers:
      x=layer(x,memory,src_mask,tgt_mask)
    return self.norm(x)
  
class Embeddings(nn.Module):
  def __init__(self,d_model,vocab):
    super(Embeddings,self).__init__()
    self.lut=nn.Embedding(vocab,d_model)
    self.d_model=d_model
  def forward(self,x):
    return self.lut(x)*math.sqrt(self.d_model)

class Generator(nn.Module):
  def __init__(self,d_model,vocab):
    super(Generator,self).__init__()
    self.proj=nn.Linear(d_model,vocab)
  def forward(self,x):
    return F.log_softmax(self.proj(x),dim=-1)

class LabelSmoothing(nn.Module):
  def __init__(self, size, padding_idx, smoothing=0.0):
    super(LabelSmoothing, self).__init__()
    # KLDivLoss expects log-probabilities for input and probabilities for target.
    # reduction='sum' means we need to normalize by batch size later if desired.
    self.criterion = nn.KLDivLoss(reduction='sum')
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.size = size # vocab_size
    # Removed self.true_dist as an instance variable to prevent graph issues.

  def forward(self, x, target):
    # x: log-probabilities from model.generator, shape (batch_size * seq_len, vocab_size)
    # target: ground truth token IDs, shape (batch_size * seq_len)

    assert x.size(1) == self.size, f"Input x size {x.size(1)} does not match vocab size {self.size}"

    # Create the smoothed distribution for the target
    # It's crucial that true_dist does NOT require gradients and is not connected to x's graph.
    true_dist = torch.zeros(x.size(), device=target.device, requires_grad=False)
    # Fill non-padding positions with smoothed value
    true_dist.fill_(self.smoothing / (self.size - 2))

    # Scatter the confidence value to the correct target classes
    true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

    # Mask out padding index in the true_dist, ensuring its probability is 0
    true_dist[:, self.padding_idx] = 0

    # For any position where the actual target token is <pad>, its entire distribution should be zeroed out.
    pad_mask = (target == self.padding_idx).unsqueeze(1)
    true_dist.masked_fill_(pad_mask, 0.0)

    # KLDivLoss expects log-probabilities for input (x) and probabilities for target (true_dist).
    # Pass the locally created true_dist directly to the criterion.
    return self.criterion(x, true_dist)