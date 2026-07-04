from tqdm.auto import tqdm
import torch

def train_model(epochs,train_dataloader,val_dataloader,model,optimizer,criterion,device):
    print("Starting training...")
    for epoch in tqdm(range(epochs)):
        model.train() # Set model to training mode
        total_loss = 0
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(input_ids)

            # For language modeling, we predict the next token given the current tokens.
            # CrossEntropyLoss expects (N, C, ...) where C is num_classes.
            # Reshape output to (batch_size * seq_len, vocab_size)
            # Reshape labels to (batch_size * seq_len)
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval() # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad(): # Disable gradient calculations during validation
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                output = model(input_ids)
                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Average Validation Loss: {avg_val_loss:.4f}")

    print("Training complete!")