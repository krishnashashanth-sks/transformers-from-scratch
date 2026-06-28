import torch

def evaluate_step_llama(model, batch, clm_criterion, device,vocab_size_llama):
  model.eval()
  with torch.no_grad():
    input_ids=batch['input_ids'].to(device)
    labels=batch['labels'].to(device)
    attention_mask=batch['attention_mask'].to(device)
    logits=model(input_ids,attention_mask=attention_mask)
    loss=clm_criterion(logits.view(-1,vocab_size_llama),labels.view(-1))
  return loss.item()