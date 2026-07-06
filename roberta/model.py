import torch.nn as nn
from layers import RoBERTaEncoder

class RoBERTaForSequenceClassification(nn.Module):
  def __init__(self,config,num_labels):
    super().__init__()
    self.num_labels=num_labels
    self.roberta=RoBERTaEncoder(config)
    self.dropout=nn.Dropout(0.1)
    self.classifier=nn.Linear(config.hidden_size,num_labels)
  def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,labels=None):
    encoder_outputs=self.roberta(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids
    )
    cls_output=encoder_outputs[:,0]
    dropout_output=self.dropout(cls_output)
    logits=self.classifier(dropout_output)
    loss=0
    if labels is not None:
      loss_fct=nn.CrossEntropyLoss()
      loss=loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
    return {"logits":logits,"loss":loss}