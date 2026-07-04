def train_model(epochs,dataloader,model,optimizer,criterion,vocab_size):
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask, target_ids) in enumerate(dataloader):
            # Move data to the appropriate device (e.g., GPU)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)

            optimizer.zero_grad() # Clear gradients

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Shift targets for next token prediction
            # The model predicts the next token based on the current sequence.
            # So, if input is [T1, T2, T3], targets should be [T2, T3, <EOS>]
            # We flatten logits and targets to use CrossEntropyLoss

            # For next token prediction, we typically want to predict target_ids[i] from logits[i-1]
            # Simplified: predict target_ids from logits

            # Reshape logits to (batch_size * sequence_length, vocab_size)
            # Reshape target_ids to (batch_size * sequence_length)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    print("Conceptual training complete.")