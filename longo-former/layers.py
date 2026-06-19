import torch
import torch.nn as nn
import torch.nn.functional as F

class LongformerSelfAttention(nn.Module):
  def __init__(self,hidden_size,num_attention_heads,attention_probs_dropout_prob,attention_window):
    super().__init__()
    if hidden_size % num_attention_heads!=0:
      raise ValueError(
          f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
      )
    self.num_attention_heads=num_attention_heads
    self.attention_window=attention_window
    self.attention_head_size=int(hidden_size/num_attention_heads)
    self.all_head_size=self.num_attention_heads*self.attention_head_size
    self.query=nn.Linear(hidden_size,self.all_head_size)
    self.key=nn.Linear(hidden_size,self.all_head_size)
    self.value=nn.Linear(hidden_size,self.all_head_size)
    self.dropout=nn.Dropout(attention_probs_dropout_prob)
    self.output = nn.Linear(hidden_size, hidden_size)

  def transpose_for_scores(self,x):
    new_x_shape=x.size()[:-1]+(self.num_attention_heads,self.attention_head_size)
    x=x.view(*new_x_shape)
    return x.permute(0,2,1,3)

  def _sliding_window_attention(self,query_layer,key_layer,value_layer,attention_mask,seq_len):
    window_size=self.attention_window
    half_window=window_size //2

    # Create additive band mask (0 inside window, -inf outside)
    indices = torch.arange(seq_len, device=query_layer.device)
    mask = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() <= half_window # (seq_len, seq_len)
    additive_band_mask = mask.unsqueeze(0).unsqueeze(0).float().masked_fill(mask == False, float('-inf')).masked_fill(mask == True, 0.0)

    attention_scores=torch.matmul(query_layer,key_layer.transpose(-1,-2))
    # Existing sanitize after matmul
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)

    attention_scores=attention_scores/(self.attention_head_size**0.5)
    # Existing sanitize after scaling
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)

    # NEW: Check after matmul and scaling
    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
      print("NAN/INF detected in attention_scores after matmul and scaling (sliding window) - AFTER SANITIZE")

    # Apply additive band mask
    attention_scores = attention_scores + additive_band_mask
    # NEW: Sanitize after additive band mask application
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
        print("NAN/INF detected in attention_scores AFTER additive_band_mask (sliding window)")

    # Apply padding attention mask if provided
    if attention_mask is not None:
        # We need to mask out columns (key dimension) in attention_scores where attention_mask is 0.
        padding_mask_for_keys = (attention_mask == 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S) boolean mask
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            print("NAN/INF detected in attention_scores BEFORE padding_mask_for_keys (sliding window) - BEFORE masked_fill") # NEW DEBUG PRINT
        attention_scores = attention_scores.masked_fill(padding_mask_for_keys, float('-inf'))
        # NEW: Sanitize immediately after padding mask application
        attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            print("NAN/INF detected in attention_scores AFTER padding_mask_for_keys (sliding window) - AFTER SANITIZE") # MODIFIED DEBUG PRINT

    # NEW: Sanitize after *all* masking additions, before clamping
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
        print("NAN/INF detected in attention_scores AFTER final nan_to_num and before clamp (sliding window)")

    # Clamp attention scores AFTER all masks have been applied to ensure finite values for softmax
    attention_scores = torch.clamp(attention_scores, min=-1e4, max=1e4)

    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
      print("!!!CRITICAL!!! NAN/INF detected in attention_scores AFTER clamp (sliding window)") # Modified print

    # Store which rows would become all -inf. These will produce NaN after softmax
    all_neg_inf_rows = (attention_scores == float('-inf')).all(dim=-1)

    attention_probs=nn.Softmax(dim=-1)(attention_scores)

    # Explicitly set attention_probs to zero for rows that were all -inf before softmax
    # This prevents nan propagation and avoids inplace modification
    attention_probs = torch.where(all_neg_inf_rows.unsqueeze(-1).expand_as(attention_probs),
                                  torch.zeros_like(attention_probs),
                                  attention_probs)

    # Existing: Aggressively sanitize attention_probs after all calculations and fixes
    attention_probs = torch.nan_to_num(attention_probs, nan=0.0, posinf=1.0, neginf=0.0)


    if torch.isnan(attention_probs).any():
      print("NAN detected in attention_probs after softmax (sliding window) - AFTER FIX ATTEMPT")

    attention_probs=self.dropout(attention_probs)
    context_layer=torch.matmul(attention_probs,value_layer)

    # Existing: Aggressively sanitize context_layer after matmul
    context_layer = torch.nan_to_num(context_layer, nan=0.0, posinf=1e4, neginf=-1e4)

    if torch.isnan(context_layer).any() or torch.isinf(context_layer).any():
      print("NAN/INF detected in context_layer (sliding window)")

    return context_layer

  def _create_sliding_window_mask(self,seq_len,device):
    # This method is not used in the current forward pass, but keeping it for completeness.
    half_window=self.attention_window//2
    mask=torch.zeros(seq_len,seq_len,device=device,dtype=torch.long)
    for i in range(seq_len):
      start=max(0,i+half_window)
      end=min(seq_len,i+half_window+1)
      mask[i,start:end]=True
    return mask.unsqueeze(0).unsqueeze(0)

  def _get_global_mask(self,global_attention_mask,seq_len,device):
    # global_attention_mask: (batch_size, seq_len) where 1 means global, 0 means local
    # Create a mask (batch_size, 1, seq_len, seq_len) where True means allowed interaction
    global_tokens = global_attention_mask.bool() # (batch_size, seq_len)

    # Expand global_tokens to match query and key dimensions for broadcasting
    query_global = global_tokens.unsqueeze(2)  # (batch_size, seq_len, 1)
    key_global = global_tokens.unsqueeze(1)    # (batch_size, 1, seq_len)

    # global_interaction_mask is True where either the query token or the key token is global
    # This broadcasts (batch_size, seq_len, 1) and (batch_size, 1, seq_len) to (batch_size, seq_len, seq_len)
    global_interaction_mask = query_global | key_global # (batch_size, seq_len, seq_len)

    # Add a head dimension (num_heads is 1 here, will broadcast later with num_attention_heads)
    return global_interaction_mask.unsqueeze(1) # (batch_size, 1, seq_len, seq_len)

  def _global_attention(self,query_layer,key_layer,value_layer,global_attention_mask,padding_attention_mask,seq_len):
    device=query_layer.device
    attention_scores=torch.matmul(query_layer,key_layer.transpose(-1,-2))
    # Existing sanitize after matmul
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)

    attention_scores=attention_scores/(self.attention_head_size**0.5)
    # Existing sanitize after scaling
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)

    # NEW: Check after matmul and scaling
    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
      print("NAN/INF detected in attention_scores after matmul and scaling (global) - AFTER SANITIZE")

    # global_interaction_mask is (batch_size, 1, seq_len, seq_len)
    global_interaction_mask=self._get_global_mask(global_attention_mask,seq_len,device)

    # Convert boolean mask to additive mask (-inf for forbidden, 0 for allowed)
    additive_global_mask = global_interaction_mask.float().masked_fill(global_interaction_mask == False, float('-inf')).masked_fill(global_interaction_mask == True, 0.0)
    attention_scores = attention_scores + additive_global_mask
    # NEW: Sanitize after additive global mask application
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
        print("NAN/INF detected in attention_scores AFTER additive_global_mask (global)")

    if padding_attention_mask is not None:
      padding_mask_for_keys = (padding_attention_mask == 0).unsqueeze(1).unsqueeze(2)
      if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
        print("NAN/INF detected in attention_scores BEFORE padding_mask_for_keys (global) - BEFORE masked_fill") # NEW DEBUG PRINT
      attention_scores = attention_scores.masked_fill(padding_mask_for_keys, float('-inf'))
      # NEW: Sanitize immediately after padding mask application
      attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
      if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
        print("NAN/INF detected in attention_scores AFTER padding_mask_for_keys (global) - AFTER SANITIZE") # MODIFIED DEBUG PRINT

    # NEW: Sanitize after *all* masking additions, before clamping
    attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
        print("NAN/INF detected in attention_scores AFTER final nan_to_num and before clamp (global)")

    # Clamp attention scores AFTER all masks have been applied to ensure finite values for softmax
    attention_scores = torch.clamp(attention_scores, min=-1e4, max=1e4)

    if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
      print("!!!CRITICAL!!! NAN/INF detected in attention_scores AFTER clamp (global)") # Modified print

    # Store which rows would become all -inf. These will produce NaN after softmax
    all_neg_inf_rows = (attention_scores == float('-inf')).all(dim=-1)

    attention_probs=nn.Softmax(dim=-1)(attention_scores)

    # Explicitly set attention_probs to zero for rows that were all -inf before softmax
    # This prevents nan propagation and avoids inplace modification
    attention_probs = torch.where(all_neg_inf_rows.unsqueeze(-1).expand_as(attention_probs),
                                  torch.zeros_like(attention_probs),
                                  attention_probs)

    # Existing: Aggressively sanitize attention_probs after all calculations and fixes
    attention_probs = torch.nan_to_num(attention_probs, nan=0.0, posinf=1.0, neginf=0.0)


    if torch.isnan(attention_probs).any():
      print("NAN detected in attention_probs after softmax (global) - AFTER FIX ATTEMPT")

    attention_probs=self.dropout(attention_probs)
    context_layer=torch.matmul(attention_probs,value_layer)

    # Existing: Aggressively sanitize context_layer after matmul
    context_layer = torch.nan_to_num(context_layer, nan=0.0, posinf=1e4, neginf=-1e4)

    if torch.isnan(context_layer).any() or torch.isinf(context_layer).any():
      print("NAN/INF detected in context_layer (global)")

    # Removed permute and view operations. This will now return (batch_size, num_heads, seq_len, head_dim)
    return context_layer

  def forward(self,hidden_states,attention_mask=None,global_attention_mask=None):
    mixed_query_layer=self.query(hidden_states)
    mixed_key_layer=self.key(hidden_states)
    mixed_value_layer=self.value(hidden_states)

    # Add nan_to_num here to sanitize outputs of linear layers (Previous Fix)
    mixed_query_layer = torch.nan_to_num(mixed_query_layer, nan=0.0, posinf=1e4, neginf=-1e4)
    mixed_key_layer = torch.nan_to_num(mixed_key_layer, nan=0.0, posinf=1e4, neginf=-1e4)
    mixed_value_layer = torch.nan_to_num(mixed_value_layer, nan=0.0, posinf=1e4, neginf=-1e4)

    query_layer=self.transpose_for_scores(mixed_query_layer)
    batch_size, num_heads, seq_len, head_dim = query_layer.size()
    key_layer=self.transpose_for_scores(mixed_key_layer)
    value_layer=self.transpose_for_scores(mixed_value_layer)

    # Existing: Aggressively sanitize query, key, value layers after transpose
    query_layer = torch.nan_to_num(query_layer, nan=0.0, posinf=1e4, neginf=-1e4)
    key_layer = torch.nan_to_num(key_layer, nan=0.0, posinf=1e4, neginf=-1e4)
    value_layer = torch.nan_to_num(value_layer, nan=0.0, posinf=1e4, neginf=-1e4)

    if torch.isnan(query_layer).any() or torch.isinf(query_layer).any():
      print("NAN/INF detected in query_layer after transpose and nan_to_num")
    if torch.isnan(key_layer).any() or torch.isinf(key_layer).any():
      print("NAN/INF detected in key_layer after transpose and nan_to_num")
    if torch.isnan(value_layer).any() or torch.isinf(value_layer).any():
      print("NAN/INF detected in value_layer after transpose and nan_to_num")


    # Calculate local attention
    local_context_layer=self._sliding_window_attention(
        query_layer,key_layer,value_layer,attention_mask,seq_len
    )

    final_context_layer=local_context_layer.clone()

    # Integrate global attention if global tokens are present
    if global_attention_mask is not None and global_attention_mask.sum()>0:
      computed_global_context=self._global_attention(
          query_layer,key_layer,value_layer,global_attention_mask,attention_mask,seq_len
      )
      if torch.isnan(computed_global_context).any() or torch.isinf(computed_global_context).any():
        print("NAN/INF detected in computed_global_context")

      # Determine which tokens are global for conditional update
      is_global_token = global_attention_mask.bool().unsqueeze(-1).unsqueeze(1) # (batch_size, 1, seq_len, 1)
      # Expand to (batch_size, num_heads, seq_len, head_dim)
      is_global_token = is_global_token.expand(-1, num_heads, -1, head_dim)

      # Update only the global tokens with global attention output
      final_context_layer = torch.where(is_global_token, computed_global_context, local_context_layer)

    # Apply query_padding_mask to the final context layer to ensure padding tokens yield zero output
    if attention_mask is not None:
        # attention_mask is (batch_size, seq_len). Expand to (batch_size, 1, seq_len, 1)
        query_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1).float()
        final_context_layer = final_context_layer * query_mask_expanded

    # Existing: Aggressively sanitize final_context_layer before reshaping and output projection
    final_context_layer = torch.nan_to_num(final_context_layer, nan=0.0, posinf=1e4, neginf=-1e4)

    if torch.isnan(final_context_layer).any() or torch.isinf(final_context_layer).any():
      print("NAN/INF detected in final_context_layer before output projection and nan_to_num")

    # Reshape final_context_layer from (batch_size, num_heads, seq_len, head_dim) to (batch_size, seq_len, all_head_size)
    # before passing to the output linear layer.
    final_context_layer = final_context_layer.permute(0, 2, 1, 3).contiguous()
    output_attention=self.output(final_context_layer.view(batch_size, seq_len, -1))

    if torch.isnan(output_attention).any() or torch.isinf(output_attention).any():
      print("NAN/INF detected in output_attention")

    return output_attention
  
class LongformerEncoderLayer(nn.Module):
  def __init__(self,hidden_size,num_attention_heads,intermediate_size,attention_probs_dropout_prob,hidden_dropout_prob,attention_window,layer_norm_eps=1e-12, layer_idx=-1):
    super().__init__()
    self.layer_idx = layer_idx # Store layer index for debugging
    self.attention=LongformerSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        attention_window=attention_window,
    )
    self.attention_output_dropout=nn.Dropout(hidden_dropout_prob)
    self.attention_layer_norm=nn.LayerNorm(hidden_size,eps=layer_norm_eps)
    self.intermediate=nn.Linear(hidden_size,intermediate_size)
    self.output=nn.Linear(intermediate_size,hidden_size)
    self.output_layer_norm=nn.LayerNorm(hidden_size,eps=layer_norm_eps)

  def forward(self,hidden_states,attention_mask=None,global_attention_mask=None):
    # Add robust check for numerical stability at the input of the encoder layer
    if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
        print(f"NAN/INF detected in hidden_states (input to EncoderLayer {self.layer_idx}). Replacing with finite values.")
        # Replace NaNs with 0, Infs with large finite values
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)

    # Store original hidden_states for residual connection before masking for attention
    residual = hidden_states

    # Create a masked version of hidden_states for attention calculation
    # This ensures that padding tokens (where attention_mask is 0) have their input features zeroed out
    # before computing queries, keys, and values. This prevents NaN propagation from NaN * 0 = NaN.
    if attention_mask is not None:
        expanded_attention_mask = attention_mask.unsqueeze(-1).float() # (batch_size, seq_len, 1)
        # Use torch.where to explicitly zero out hidden states for padding tokens
        hidden_states_for_attention = torch.where(expanded_attention_mask.bool(), hidden_states, torch.zeros_like(hidden_states))
    else:
        hidden_states_for_attention = hidden_states

    self_attention_outputs=self.attention(
        hidden_states_for_attention, # Pass the masked hidden_states to attention
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
    )
    attention_output=self.attention_output_dropout(self_attention_outputs)

    # Apply layer normalization with residual connection.
    # We add the attention output (which is zero for padding queries) to the original `residual` (hidden_states before attention masking).
    hidden_states = self.attention_layer_norm(residual + attention_output)

    # Feed-forward network part
    residual_ffn = hidden_states # Store for residual connection from attention block
    intermediate_output=self.intermediate(hidden_states)
    intermediate_output=F.gelu(intermediate_output)
    layer_output=self.output(intermediate_output)

    # Apply layer normalization with residual connection for FFN
    return self.output_layer_norm(residual_ffn + layer_output)
