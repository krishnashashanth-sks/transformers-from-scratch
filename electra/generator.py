from transformers import BertConfig, BertForMaskedLM
import torch.nn as nn

class ElectraGenerator(nn.Module):
  def __init__(self,vocab_size,hidden_size,num_attention_heads,num_hidden_layers,intermediate_size):
    super(ElectraGenerator,self).__init__()
    config=BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        is_decoder=False
    )
    self.bert_model=BertForMaskedLM(config)
  def forward(self,input_ids,attention_mask=None,token_type_ids=None,labels=None):
    outputs=self.bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels
    )
    return outputs.logits