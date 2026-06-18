import torch
from tqdm.notebook import tqdm # For progress bar
import time

def train_model(model, dataloader, loss_function, optimizer, scheduler, device, num_epochs):
    print("Starting training...")
    history = {'train_loss': [], 'train_acc': [], 'lr': []}

    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train() # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Train for one epoch (logic previously in train_one_epoch)
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            inputs = inputs.to(device) # (batch_size, C, T, H, W)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs) # outputs shape: (batch_size, num_classes)

            # Calculate loss
            loss = loss_function(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item() * inputs.size(0) # inputs.size(0) is batch_size
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        # Update learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        end_time = time.time()
        epoch_duration = end_time - start_time

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_accuracy)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1}/{num_epochs} | ")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | ")
        print(f"  Learning Rate: {current_lr:.6f} | Time: {epoch_duration:.2f}s")
        print("-" * 50)

    print("Training complete.")
    return history