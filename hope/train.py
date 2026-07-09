import torch

def train_model(num_epochs,num_batches_per_epoch,vocab_size,batch_size,seq_len,optimizer,hope_model,loss_function):
    print("Starting full training loop...")

    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch_idx in range(num_batches_per_epoch):
            # 1. Simulate input_ids and target_ids for a batch
            input_ids_batch = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
            target_ids_batch = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

            # 2. Zero the gradients
            optimizer.zero_grad()

            # 3. Forward pass through the HOPE model
            # Fast weights are generated and applied dynamically inside this call
            output_logits, _ = hope_model(input_ids_batch)

            # Reshape for CrossEntropyLoss
            loss_input = output_logits.view(-1, vocab_size)
            loss_target = target_ids_batch.view(-1)

            # 4. Calculate loss
            loss = loss_function(loss_input, loss_target)

            # 5. Backward pass
            loss.backward()

            # 6. Update slow weights
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy for this batch
            predictions = torch.argmax(output_logits, dim=-1)
            total_correct += (predictions == target_ids_batch).sum().item()
            total_tokens += (batch_size * seq_len)

        avg_loss = total_loss / num_batches_per_epoch
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

    print("Full training loop completed. Slow weights have been iteratively updated.")
