from tqdm.auto import tqdm

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train() # Set the model to training mode
    running_loss = 0.0
    # Use tqdm for a progress bar
    for batch in tqdm(dataloader, desc="Training Epoch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # Zero out gradients

        outputs = model(input_ids, mask=attention_mask.unsqueeze(1).unsqueeze(2))

        # Reshape outputs and labels for CrossEntropyLoss
        # outputs: (batch_size, seq_len, vocab_size)
        # labels: (batch_size, seq_len)
        outputs = outputs.view(-1, outputs.size(-1)) # (batch_size * seq_len, vocab_size)
        labels = labels.view(-1) # (batch_size * seq_len)

        loss = loss_fn(outputs, labels)

        loss.backward() # Backward pass
        optimizer.step() # Update model parameters

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

print("train_epoch function defined.")