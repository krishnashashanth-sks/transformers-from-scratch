import torch.nn as nn
import torch
from layers import StarCoderV2Embeddings,StarCoderV2Block,RMSNorm

class StarCoderV2Model(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config=config
    self.embeddings=StarCoderV2Embeddings(config)
    self.h=nn.ModuleList([StarCoderV2Block(config) for _ in range(config.num_hidden_layers)])
    self.norm=RMSNorm(config.hidden_size,eps=config.norm_epsilon)
  def forward(self,input_ids,attention_mask=None,position_ids=None):
    batch_size,seq_len=input_ids.size()
    if position_ids is None:
      position_ids=torch.arange(seq_len,dtype=torch.long,device=input_ids.device)
      position_ids=position_ids.unsqueeze(0).expand_as(input_ids)
    if attention_mask is not None:
      attention_mask = attention_mask.to(torch.float)
      causal_mask_base=torch.triu(torch.full((seq_len,seq_len),-torch.inf, device=attention_mask.device, dtype=attention_mask.dtype),diagonal=1)
      expanded_attn_mask=(attention_mask[:,None,:]).repeat(1,seq_len,1)
      combined_mask=expanded_attn_mask.masked_fill(expanded_attn_mask==0,-torch.inf)+causal_mask_base
      attention_mask = combined_mask.unsqueeze(1)
    hidden_states=self.embeddings(input_ids)
    for layer_module in self.h:
      hidden_states=layer_module(hidden_states,attention_mask,position_ids)
    return self.norm(hidden_states)


class StarCoderV2ForCausalLM(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.transformer=StarCoderV2Model(config)
    self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
  def forward(self,input_ids,attention_mask=None,position_ids=None,labels=None):
    transformer_output=self.transformer(input_ids,attention_mask,position_ids)
    hidden_states=transformer_output
    lm_logits=self.lm_head(hidden_states)
    loss=None
    if labels is not None:
      shift_logits=lm_logits[...,:-1,:].contiguous()
      shift_labels=labels[...,1:].contiguous()
      loss_fct=nn.CrossEntropyLoss()
      loss=loss_fct(shift_logits.view(-1,shift_logits.size(-1)),shift_labels.view(-1))
    return {"logits":lm_logits,"loss":loss}
