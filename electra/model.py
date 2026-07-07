import torch.nn as nn
from discriminator import ElectraDiscriminator
from generator import ElectraGenerator
import torch

class ElectraModel(nn.Module):
  def __init__(self,generator_config,discriminator_config,mask_prob,replace_prob,mask_token_id,pad_token_id):
    super(ElectraModel,self).__init__()
    self.generator=ElectraGenerator(**generator_config)
    self.discriminator=ElectraDiscriminator(**discriminator_config)
    self.mask_prob=mask_prob
    self.replace_prob=replace_prob
    self.mask_token_id=mask_token_id
    self.pad_token_id = pad_token_id # Corrected line: assign pad_token_id
  def _apply_masking_and_replacement(self,input_ids,original_labels,attention_mask,generator_logits):
    batch_size,seq_len=input_ids.shape
    device=input_ids.device
    can_mask=(attention_mask==1) & (input_ids != self.pad_token_id)
    num_to_mask=(can_mask.sum(dim=1)*self.mask_prob).long()
    gen_masked_input_ids=input_ids.clone()
    gen_labels=original_labels.clone()
    masked_indices=torch.zeros((batch_size,seq_len),dtype=torch.bool,device=device)
    for i in range(batch_size):
      if num_to_mask[i]>0:
        indices_to_mask=torch.randperm(can_mask[i].sum())[:num_to_mask[i]]
        actual_indices=torch.where(can_mask[i])[0][indices_to_mask]
        masked_indices[i,actual_indices]=True
    gen_masked_input_ids[masked_indices] = self.mask_token_id
    gen_labels[~masked_indices] = -100
    predicted_tokens=torch.argmax(generator_logits,dim=-1)
    corrupted_input_ids=input_ids.clone()
    discriminator_labels=torch.zeros_like(input_ids,dtype=torch.float,device=device)
    mismatched_replacements=(predicted_tokens!= original_labels) & masked_indices
    corrupted_input_ids[mismatched_replacements]=predicted_tokens[mismatched_replacements]
    discriminator_labels[mismatched_replacements]=1.0
    disc_input=input_ids.clone()
    disc_labels=torch.zeros_like(input_ids,dtype=torch.float,device=device)
    for i in range(batch_size):
      for j in range(seq_len):
        if masked_indices[i,j]:
          if predicted_tokens[i,j]!=original_labels[i,j]:
            disc_input[i,j]=predicted_tokens[i,j]
            disc_labels[i,j]=1.0
          else:
            disc_input[i,j] = original_labels[i,j]
    disc_labels=disc_labels*attention_mask
    return gen_masked_input_ids,gen_labels,disc_input,disc_labels
  def forward(self,input_ids,attention_mask=None,token_type_ids=None,original_labels=None):
    temp_gen_output_logits=self.generator(input_ids,attention_mask,token_type_ids) # Generator sees unmasked input to give predictions
    gen_masked_input_ids,gen_labels,corrupted_input_ids,discriminator_labels=\
    self._apply_masking_and_replacement(input_ids,original_labels,attention_mask,temp_gen_output_logits)
    generator_logits=self.generator(gen_masked_input_ids,attention_mask,token_type_ids,labels=gen_labels)
    discriminator_logits=self.discriminator(corrupted_input_ids,attention_mask,token_type_ids)
    return generator_logits,gen_labels,discriminator_logits,discriminator_labels