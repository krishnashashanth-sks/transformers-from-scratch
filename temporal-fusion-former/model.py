import torch.nn as nn
import torch
from layers import StaticCovariateEncoder,DynamicCovariateEncoder,GRN,MultiHeadAttention

class TemporalFusionTransformer(nn.Module):
  def __init__(self,
               hidden_size:int,
               num_heads:int,
               num_gru_layers:int,
               dropout_rate:float,
               output_quantiles:list,
               static_categorical_sizes:list,
               static_real_size:int,
               known_dynamic_categorical_sizes:list,
               known_dynamic_real_size:int,
               unknown_dynamic_categorical_sizes:list,
               unknown_dynamic_real_size:int, # This now refers to EXPLICIT unknown dynamic real features
               encoder_length:int,
               decoder_length:int
               ):
      super(TemporalFusionTransformer,self).__init__()
      self.hidden_size=hidden_size
      self.num_heads=num_heads
      self.num_gru_layers=num_gru_layers # FIX: Removed extraneous comma
      self.dropout_rate=dropout_rate
      self.output_quantiles=output_quantiles
      self.encoder_length=encoder_length
      self.decoder_length=decoder_length
      self.static_encoder=StaticCovariateEncoder(
          input_categorical_sizes=static_categorical_sizes,
          input_real_size=static_real_size,
          hidden_size=hidden_size,
          output_size=hidden_size,
          dropout_rate=dropout_rate
      )
      self.dynamic_encoder=DynamicCovariateEncoder(
          known_categorical_sizes=known_dynamic_categorical_sizes,
          known_real_size=known_dynamic_real_size,
          unknown_categorical_sizes=unknown_dynamic_categorical_sizes,
          unknown_real_size=(unknown_dynamic_real_size + 1),
          hidden_size=hidden_size,
          num_gru_layers=num_gru_layers,
          dropout_rate=dropout_rate
      )
      self.pre_attention_grn=GRN(
          input_size=hidden_size+hidden_size,
          hidden_size=hidden_size,
          output_size=hidden_size,
          dropout_rate=dropout_rate
      )
      self.attention=MultiHeadAttention(
          embed_dim=hidden_size,
          num_heads=num_heads,
          dropout_rate=dropout_rate
      )
      self.post_attention_grn=GRN(
          input_size=hidden_size,
          hidden_size=hidden_size,
          output_size=hidden_size,
          dropout_rate=dropout_rate
      )
      self.attention_norm=nn.LayerNorm(hidden_size)
      self.feed_forward_grn=GRN(
          input_size=hidden_size,
          hidden_size=hidden_size,
          output_size=hidden_size,
          dropout_rate=dropout_rate
      )
      self.feed_forward_norm=nn.LayerNorm(hidden_size)
      self.final_combiner_grn=GRN(
          input_size=hidden_size*2,
          hidden_size=hidden_size,
          output_size=hidden_size,
          dropout_rate=dropout_rate
      )
      self.output_layer=nn.Linear(hidden_size,len(output_quantiles))
  def forward(self,
              static_categorical_data:list,
              static_real_data:torch.Tensor,
              historical_known_categorical_data:list,
              historical_known_real_data:torch.Tensor,
              historical_unknown_categorical_data:list,
              historical_unknown_real_data:torch.Tensor,
              future_known_categorical_data:list,
              future_known_real_data:torch.Tensor
              ):
    batch_size=historical_known_real_data.shape[0]

    static_context_vector=self.static_encoder(
        static_categorical_data,
        static_real_data
    )

    lstm_output,processed_future_features=self.dynamic_encoder(
        historical_known_categorical_data,
        historical_known_real_data,
        historical_unknown_categorical_data,
        historical_unknown_real_data,
        future_known_categorical_data,
        future_known_real_data,
        static_context_vector
    )

    static_context_expanded=static_context_vector.unsqueeze(1).expand(-1,self.encoder_length,-1)

    pre_attention_input=torch.cat([lstm_output,static_context_expanded],dim=-1)

    query_key_value=self.pre_attention_grn(pre_attention_input)

    attn_output,attn_weights=self.attention(query_key_value,query_key_value,query_key_value)
    # FIX: Correct residual connection for attention block
    attn_output=self.attention_norm(lstm_output + attn_output)

    ff_output=self.feed_forward_grn(attn_output)

    encoder_output_with_attention=self.feed_forward_norm(attn_output+ff_output)

    historical_context_for_prediction=encoder_output_with_attention[:,-1,:].unsqueeze(1)

    historical_context_for_prediction=historical_context_for_prediction.expand(-1,self.decoder_length,-1)

    combined_decoder_input=torch.cat([
        historical_context_for_prediction,
        processed_future_features
    ],dim=-1)

    final_decoder_output=self.final_combiner_grn(combined_decoder_input)

    return self.output_layer(final_decoder_output)