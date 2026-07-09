import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    # 2. Initialize an optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 3. Initialize a loss function
    # ignore_index=0 because 0 is used for padding in target sequences
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # 4. Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    model.to(device)

    print(f"Starting training for {num_epochs} epochs...")

    # 5. Implement the main training loop
    for epoch in range(num_epochs):
        model.train() # a. Set the model to training mode
        current_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for inputs, targets in train_bar:
            inputs = inputs.to(device) # c. Move inputs to device
            targets = targets.to(device) # c. Move targets to device

            optimizer.zero_grad() # d. Zero out the optimizer's gradients

            outputs = model(inputs) # e. Perform a forward pass

            # f. Reshape outputs and targets for CrossEntropyLoss
            # outputs: (batch_size, sequence_length, vocab_size)
            # targets: (batch_size, sequence_length)
            outputs = outputs.view(-1, outputs.size(-1)) # (batch_size * sequence_length, vocab_size)
            targets = targets.view(-1) # (batch_size * sequence_length)

            loss = loss_fn(outputs, targets) # g. Calculate the loss
            loss.backward() # h. Perform a backward pass
            optimizer.step() # i. Update model weights

            current_train_loss += loss.item() # j. Accumulate training loss
            train_bar.set_postfix(loss=current_train_loss / (train_bar.n + 1))

        avg_train_loss = current_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 6. Implement a validation loop
        model.eval() # a. Set the model to evaluation mode
        current_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad(): # b. Disable gradient calculations
            for inputs, targets in val_bar:
                inputs = inputs.to(device) # d. Move inputs to device
                targets = targets.to(device) # d. Move targets to device

                outputs = model(inputs) # e. Perform a forward pass

                # f. Reshape outputs and targets
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                val_loss = loss_fn(outputs, targets) # g. Calculate the validation_loss
                current_val_loss += val_loss.item() # h. Accumulate validation loss
                val_bar.set_postfix(loss=current_val_loss / (val_bar.n + 1))

        avg_val_loss = current_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 7. Print the training and validation loss
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Training complete.")
    return train_losses, val_losses # 8. Return the lists of training and validation losses
