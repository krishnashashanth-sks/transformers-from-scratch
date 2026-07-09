from tqdm.auto import tqdm

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train() # Set the model to training mode
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids)

        # Reshape outputs and labels for CrossEntropyLoss
        # outputs: (batch_size, sequence_length, vocab_size)
        # labels: (batch_size, sequence_length)
        reshaped_outputs = outputs.view(-1, outputs.size(-1)) # (batch_size * sequence_length, vocab_size)
        reshaped_labels = labels.view(-1) # (batch_size * sequence_length)

        # Calculate loss
        loss = loss_fn(reshaped_outputs, reshaped_labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

