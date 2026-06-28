def train_step(model,batch,mlm_criterion,nsp_criterion,optimizer,device,vocab_size):
  input_ids=batch['input_ids']
  segment_ids=batch['segment_ids'].to(device)
  attention_mask=batch['attention_mask'].to(device)
  mlm_labels=batch['mlm_labels'].to(device)
  nsp_labels=batch['nsp_labels'].squeeze(1).to(device)
  optimizer.zero_grad()
  mlm_prediction_scores,nsp_prediction_scores=model(
    input_ids,
    segment_ids,
    attention_mask.unsqueeze(1).unsqueeze(2)
  )
  mlm_loss=mlm_criterion(
      mlm_prediction_scores.view(-1,vocab_size),
      mlm_labels.view(-1)
  )
  nsp_loss=nsp_criterion(
      nsp_prediction_scores,
      nsp_labels
  )
  total_loss=mlm_loss+nsp_loss
  total_loss.backward()
  optimizer.step()
  return total_loss.item(), mlm_loss.item(), nsp_loss.item()