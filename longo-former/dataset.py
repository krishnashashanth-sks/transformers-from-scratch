import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
  def __init__(self,texts,labels,tokenizer,max_length):
    self.texts=texts
    self.labels=labels
    self.tokenizer=tokenizer
    self.max_length=max_length
    self.input_ids=[]
    self.attention_masks=[]
    self.token_type_ids=[]
    self.global_attention_masks=[]
    for text in self.texts:
      encoded_input=self.tokenizer.encode(text,max_length=self.max_length)
      self.input_ids.append(encoded_input['input_ids'])
      self.attention_masks.append(encoded_input['attention_mask'])
      self.token_type_ids.append(encoded_input['token_type_ids'])
      self.global_attention_masks.append(encoded_input['global_attention_mask'])
  def __len__(self):
    return len(self.labels)
  def __getitem__(self,idx):
    return {
        "input_ids":self.input_ids[idx],
        "attention_mask":self.attention_masks[idx],
        "token_type_ids":self.token_type_ids[idx],
        "global_attention_mask":self.global_attention_masks[idx],
        'labels':torch.tensor(self.labels[idx],dtype=torch.long)
    }
