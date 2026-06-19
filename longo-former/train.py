import torch

def train_step(model,batch,optimizer,loss_fn,device):
  model.train()
  input_ids=batch['input_ids'].to(device)
  attention_mask=batch['attention_mask'].to(device)
  token_type_ids=batch['token_type_ids'].to(device) # Corrected: Removed duplicate assignment
  global_attention_mask=batch['global_attention_mask'].to(device)
  labels=batch['labels'].to(device)
  optimizer.zero_grad()
  logits=model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,global_attention_mask=global_attention_mask)
  loss=loss_fn(logits,labels)
  loss.backward()
  optimizer.step()
  return loss.item()

def eval_step(model,batch,loss_fn,device):
  model.eval()
  input_ids = batch['input_ids'].to(device)
  attention_mask = batch['attention_mask'].to(device)
  token_type_ids = batch['token_type_ids'].to(device) # Corrected typo: token_typed_ids -> token_type_ids
  global_attention_mask = batch['global_attention_mask'].to(device)
  labels = batch['labels'].to(device)
  with torch.no_grad():
    logits=model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,global_attention_mask=global_attention_mask)
    loss=loss_fn(logits,labels)
    predictions=torch.argmax(logits,dim=-1)
    accuracy=(predictions==labels).float().mean()
  return loss.item(), accuracy.item()
def train(num_epochs,model,dataloader,optimizer,loss_fn,device):
  # Initialize lists to store metrics
    train_losses = []
    eval_losses = []
    eval_accuracies = []

    # Best evaluation loss tracker for optional model saving
    best_eval_loss = float('inf')
    for epoch in range(num_epochs):
        # Training Phase
        model.train() # Set model to training mode
        total_train_loss = 0
        for batch_idx, batch in enumerate(dataloader): # Use dataloader as training dataloader
            loss = train_step(model, batch, optimizer, loss_fn, device)
            total_train_loss += loss

        avg_train_loss = total_train_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

        # Evaluation Phase
        model.eval() # Set model to evaluation mode
        total_eval_loss = 0
        total_eval_accuracy = 0

        # Use the same dataloader for simplicity as validation dataloader
        for batch_idx, batch in enumerate(dataloader):
            loss, accuracy = eval_step(model, batch, loss_fn, device)
            total_eval_loss += loss
            total_eval_accuracy += accuracy

        avg_eval_loss = total_eval_loss / len(dataloader)
        avg_eval_accuracy = total_eval_accuracy / len(dataloader)
        eval_losses.append(avg_eval_loss)
        eval_accuracies.append(avg_eval_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Evaluation Loss: {avg_eval_loss:.4f}, Evaluation Accuracy: {avg_eval_accuracy:.4f}")

        # Optional: Save the model if it's the best performing so far
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            # torch.save(model.state_dict(), 'best_model.pth')
            # print("Saved best model with evaluation loss: {best_eval_loss:.4f}")
    return train_losses,eval_losses,eval_accuracies