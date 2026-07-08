# --- Basic Training Loop ---
from tqdm.auto import tqdm
import torch

def train_model(num_epochs,model,train_dataloader,test_dataloader,optimizer,criterion,device):
    print("Starting training...")
    for epoch in tqdm(range(num_epochs)):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Optimize weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 100 == 99: # Print every 100 batches
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}, Train Acc: {100 * correct_train / total_train:.2f}%")
                running_loss = 0.0
                correct_train = 0
                total_train = 0

        print(f"Epoch {epoch + 1} finished.")

        # --- Evaluation ---
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc=f"Epoch {epoch+1} Evaluation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

    print("Training complete!")