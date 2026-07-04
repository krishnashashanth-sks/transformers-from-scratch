def train_model(num_epochs,dataloader,model,loss_fn,optimizer,scheduler,device):
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Use tqdm for a progress bar during training
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            # Move tensors to the appropriate device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Shift input_ids to create labels for language modeling
            # labels are input_ids shifted by one position to the left
            labels = input_ids.clone() # Make a copy to avoid modifying original input_ids
            # Target for token at position i is the token at position i+1
            labels = labels[:, 1:].contiguous() # All tokens except the first
            input_ids = input_ids[:, :-1].contiguous() # All tokens except the last
            attention_mask = attention_mask[:, :-1].contiguous() # Match attention mask to input_ids

            # Perform a forward pass
            logits = model(input_ids, attention_mask=attention_mask) # logits shape: (batch_size, seq_len-1, vocab_size)

            # Calculate the loss
            # Reshape logits to (batch_size * (seq_len-1), vocab_size)
            # Reshape labels to (batch_size * (seq_len-1))
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Perform backpropagation
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update model parameters
            scheduler.step()       # Update learning rate

            total_loss += loss.item()

            if batch_idx % 10 == 0: # Print loss every 10 batches
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    print("Training complete.")