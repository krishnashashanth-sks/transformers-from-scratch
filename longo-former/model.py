import torch.nn as nn
from layers import *

class LongformerModel(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config=config
    self.word_embeddings=nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=0)
    self.position_embeddings=nn.Embedding(config.max_position_embeddings,config.hidden_size)
    self.token_type_embeddings=nn.Embedding(config.type_vocab_size,config.hidden_size)
    self.layer_norm=nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
    self.dropout=nn.Dropout(config.hidden_dropout_prob)
    self.encoder=nn.ModuleList([
        LongformerEncoderLayer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_window=config.attention_window,
            layer_norm_eps=config.layer_norm_eps,
            layer_idx=i, # Pass layer index
        )
        for i in range(config.num_hidden_layers)
    ])
    self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    self.apply(self._init_weights)
  def _init_weights(self,module):
    if isinstance(module,(nn.Linear,nn.Embedding)):
      module.weight.data.normal_(mean=0.0,std=self.config.initializer_range)
    elif isinstance(module,nn.LayerNorm):
      module.weight.data.fill_(1.0)
    if isinstance(module,nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
  def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,global_attention_mask=None):
    if input_ids is not None:
      input_shape=input_ids.size()
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")
    seq_len=input_shape[1]
    device=input_ids.device
    if attention_mask is None:
        attention_mask=torch.ones(input_shape,device=device)
    if token_type_ids is None:
        token_type_ids=torch.zeros(input_shape,dtype=torch.long,device=device)
    if position_ids is None:
      position_ids=torch.arange(seq_len,dtype=torch.long,device=device)
      position_ids=position_ids.unsqueeze(0).expand(input_shape)
    word_embeddings=self.word_embeddings(input_ids)
    position_embeddings=self.position_embeddings(position_ids)
    token_type_embeddings=self.token_type_embeddings(token_type_ids)
    embeddings=word_embeddings+position_embeddings+token_type_embeddings
    embeddings=self.layer_norm(embeddings)
    embeddings=self.dropout(embeddings)
    hidden_states=embeddings
    for i,layer_module in enumerate(self.encoder):
      layer_outputs=layer_module(
          hidden_states,
          attention_mask=attention_mask,
          global_attention_mask=global_attention_mask
      )
      hidden_states=layer_outputs
    pooled_output = hidden_states[:, 0]
    pooled_output = self.classifier_dropout(pooled_output)
    logits = self.classifier(pooled_output)
    return logits