import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """Custom PyTorch module for multi-head self-attention."""
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1) # Using a common dropout probability

        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, sequence_length, head_dim)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # Apply the attention mask (e.g., for padding tokens)
            # Mask values are -10000.0 or -inf for masked positions
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out(context_layer)

        return output

class FeedForwardNetwork(nn.Module):
    """Custom PyTorch module for the position-wise feed-forward network."""
    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.GELU()
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1) # Using a common dropout probability

    def forward(self, hidden_states):
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerEncoderLayer(nn.Module):
    """Custom PyTorch module for a single Transformer Encoder Layer."""
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.attention_dropout = nn.Dropout(0.1) # Using a common dropout probability
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.feed_forward = FeedForwardNetwork(config)
        self.output_dropout = nn.Dropout(0.1) # Using a common dropout probability
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention Block
        # Pre-normalization
        attention_output = self.attention_layer_norm(hidden_states)
        attention_output = self.attention(attention_output, attention_mask)
        attention_output = self.attention_dropout(attention_output)
        # Residual connection
        hidden_states = hidden_states + attention_output

        # Feed-Forward Block
        # Pre-normalization
        ffn_output = self.output_layer_norm(hidden_states)
        ffn_output = self.feed_forward(ffn_output)
        ffn_output = self.output_dropout(ffn_output)
        # Residual connection
        hidden_states = hidden_states + ffn_output

        return hidden_states
    
class TransformerEncoder(nn.Module):
    """Custom PyTorch module for the Transformer Encoder."""
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states
    
class RoBERTaEmbeddings(nn.Module):
    """Custom PyTorch module for RoBERTa-style input embeddings."""
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1) # RoBERTa uses padding_idx=1
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # RoBERTa often omits token_type_embeddings or sets it to 1 type, but for generality,
        # we include it as an option, assuming 2 token types.
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        seq_length = input_ids.size(1)
        device = input_ids.device

        if position_ids is None:
            # RoBERTa's position_ids start at 2 for '<s>' and '</s>' tokens.
            # The 0 and 1 positions are reserved for padding and '<s>' respectively.
            position_ids = torch.arange(2, seq_length + 2, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class RoBERTaEncoder(nn.Module):
    """Custom PyTorch module for the full RoBERTa Encoder."""
    def __init__(self, config):
        super().__init__()
        self.embeddings = RoBERTaEmbeddings(config)
        self.encoder = TransformerEncoder(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        # Create extended attention mask
        if attention_mask is not None:
            # (batch_size, 1, 1, sequence_length)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask to float and set values to -10000.0 or 0.0
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)

        return encoder_outputs
