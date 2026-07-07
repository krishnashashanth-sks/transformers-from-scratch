import torch.nn as nn
from transformers import BertConfig, BertModel

class ElectraDiscriminator(nn.Module):
  def __init__(self,vocab_size,hidden_size,num_attention_heads,num_hidden_layers,intermediate_size):
    super(ElectraDiscriminator,self).__init__()
    config=BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size
    )
    self.bert_model=BertModel(config)
    self.classifier=nn.Linear(hidden_size,1)
  def forward(self,input_ids,attention_mask=None,token_type_ids=None):
    outputs=self.bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    sequence_output=outputs.last_hidden_state # Corrected from last_hidden_size to last_hidden_state
    logits=self.classifier(sequence_output)
    return logits.squeeze(-1)