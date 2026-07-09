import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        token_embed = self.token_embeddings(input_ids)
        pos_embed = self.position_embeddings(positions)

        # Add positional embeddings to token embeddings
        embeddings = token_embed + pos_embed
        return embeddings
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        # These are the slow weights for Q, K, V projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, fast_weights_q=None, fast_weights_k=None, fast_weights_v=None):
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # Apply fast weights if provided (e.g., additive modulation to Q, K, V outputs)
        if fast_weights_q is not None: # fast_weights_q shape: (batch_size, seq_len, num_heads, d_k)
            Q = Q + fast_weights_q
        if fast_weights_k is not None:
            K = K + fast_weights_k
        if fast_weights_v is not None:
            V = V + fast_weights_v

        # Transpose for attention calculation: (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        return output
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # Slow weights
        self.linear2 = nn.Linear(d_ff, d_model) # Slow weights

    def forward(self, x, fast_bias1=None, fast_scale1=None, fast_bias2=None, fast_scale2=None):
        # Apply fast weights if provided (e.g., additive bias or scaling)
        h = self.linear1(x)
        if fast_bias1 is not None: # fast_bias1 shape: (batch_size, seq_len, d_ff)
            h = h + fast_bias1
        if fast_scale1 is not None: # fast_scale1 shape: (batch_size, seq_len, d_ff)
            h = h * fast_scale1
        h = F.relu(h)

        output = self.linear2(h)
        if fast_bias2 is not None: # fast_bias2 shape: (batch_size, seq_len, d_model)
            output = output + fast_bias2
        if fast_scale2 is not None: # fast_scale2 shape: (batch_size, seq_len, d_model)
            output = output * fast_scale2
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model) # Slow weights
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model) # Slow weights
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, fast_weights_attn_q=None, fast_weights_attn_k=None, fast_weights_attn_v=None,
                fast_bias_ffn1=None, fast_scale_ffn1=None, fast_bias_ffn2=None, fast_scale_ffn2=None):

        # Self-Attention sub-layer
        attn_output = self.attention(x, fast_weights_q=fast_weights_attn_q,
                                     fast_weights_k=fast_weights_attn_k,
                                     fast_weights_v=fast_weights_attn_v)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-Forward sub-layer
        ffn_output = self.ffn(x, fast_bias1=fast_bias_ffn1, fast_scale1=fast_scale_ffn1,
                              fast_bias2=fast_bias_ffn2, fast_scale2=fast_scale_ffn2)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x

class ContextEncoder(nn.Module):
  def __init__(self,d_model,num_heads,context_dim):
    super().__init__()
    self.d_model=d_model
    self.num_heads=num_heads
    self.context_dim=context_dim
    input_to_aggregator_dim=d_model*2
    self.aggregator_fnn=nn.Sequential(
        nn.Linear(input_to_aggregator_dim,d_model),
        nn.ReLU(),
        nn.Linear(d_model,context_dim)
    )
  def forward(self,current_embedding,block_output_x):
    combined_context=torch.cat((current_embedding,block_output_x),dim=-1)
    return self.aggregator_fnn(combined_context)
  
class Hypernetwork(nn.Module):
    def __init__(self, context_dim, d_model, num_heads, d_ff):
        super().__init__()
        self.context_dim = context_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff

        # Generating Multi-Head Attention Weights (output should be (batch, seq_len, num_heads, d_k))
        # So, the last dim of linear layer should be (num_heads * d_k)
        self.gen_attn_q = nn.Linear(context_dim, self.num_heads * self.d_k)
        self.gen_attn_k = nn.Linear(context_dim, self.num_heads * self.d_k)
        self.gen_attn_v = nn.Linear(context_dim, self.num_heads * self.d_k)

        # Generating Feed-Forward Network (FFN) Parameters
        # These are additive/multiplicative biases/scales, should be (batch, seq_len, dim)
        self.gen_ffn_b1 = nn.Linear(context_dim, d_ff)
        self.gen_ffn_s1 = nn.Linear(context_dim, d_ff)
        self.gen_ffn_b2 = nn.Linear(context_dim, d_model) # Projects back to d_model
        self.gen_ffn_s2 = nn.Linear(context_dim, d_model) # Projects back to d_model

    def forward(self, context_vector):
        """
        Args:
            context_vector: Tensor of shape (batch_size, seq_len, context_dim)
        Returns:
            Dictionary of 'fast weights' used to modulate a target network.
        """
        batch_size, seq_len, _ = context_vector.size() # Correctly get batch_size and seq_len

        # Generate Attention Weights
        # Shape after linear: (batch_size, seq_len, num_heads * d_k)
        # Reshape to: (batch_size, seq_len, num_heads, d_k)
        fast_weights_attn_q = self.gen_attn_q(context_vector).view(
            batch_size, seq_len, self.num_heads, self.d_k
        )
        fast_weights_attn_k = self.gen_attn_k(context_vector).view(
            batch_size, seq_len, self.num_heads, self.d_k
        )
        fast_weights_attn_v = self.gen_attn_v(context_vector).view(
            batch_size, seq_len, self.num_heads, self.d_k
        )

        # Generate FFN Bias and Scales
        # No unsqueeze(1) needed if context_vector already has seq_len dimension
        fast_bias_ffn1  = self.gen_ffn_b1(context_vector) # Shape: (batch_size, seq_len, d_ff)
        fast_scale_ffn1 = torch.sigmoid(self.gen_ffn_s1(context_vector)) # Shape: (batch_size, seq_len, d_ff)

        fast_bias_ffn2  = self.gen_ffn_b2(context_vector) # Shape: (batch_size, seq_len, d_model)
        fast_scale_ffn2 = torch.sigmoid(self.gen_ffn_s2(context_vector)) # Shape: (batch_size, seq_len, d_model)

        return {
            'attn_q': fast_weights_attn_q,
            'attn_k': fast_weights_attn_k,
            'attn_v': fast_weights_attn_v,
            'ffn_b1': fast_bias_ffn1,
            'ffn_s1': fast_scale_ffn1,
            'ffn_b2': fast_bias_ffn2,
            'ffn_s2': fast_scale_ffn2,
        }
