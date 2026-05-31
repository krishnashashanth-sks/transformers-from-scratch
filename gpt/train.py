from tqdm.auto implrt tqdm

def train(gpt_model,dataloader,optimizer,criterion,num_epochs,device):
  train_losses=[]
  for epoch in tqdm(range(num_epochs), desc="Epochs"): # tqdm for epoch progress
    # 5. Set the model to training mode
    gpt_model.train()
    epoch_loss = 0

    # 6. Iterate through the dataloader
    for batch_idx, (input_batch, target_batch) in enumerate(tqdm(dataloader, desc=f"Batch (Epoch {epoch+1})", leave=False)): # tqdm for batch progress
        # 7. Move data to the device
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # 8. Zero out the gradients
        optimizer.zero_grad()

        # 9. Perform a forward pass
        output_logits = gpt_model(input_batch)

        # 10. Reshape output_logits and target_batch for CrossEntropyLoss
        # output_logits shape: (batch_size, sequence_length, vocab_size_char)
        # target_batch shape: (batch_size, sequence_length)
        loss_logits = output_logits.view(-1, vocab_size_char)
        loss_targets = target_batch.view(-1)

        # 11. Calculate the loss
        loss = criterion(loss_logits, loss_targets)

        # 12. Perform a backward pass
        loss.backward()

        # 13. Update the model's parameters
        optimizer.step()

        # 14. Append the loss.item() to the train_losses list
        train_losses.append(loss.item())
        epoch_loss += loss.item()

        # 15. Periodically print the current epoch, batch number, and loss
        if batch_idx % 1000 == 0: # Print every 1000 batches
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    print(f"End of Epoch {epoch+1}, Average Loss: {epoch_loss / len(dataloader):.4f}")
    return train_losses
