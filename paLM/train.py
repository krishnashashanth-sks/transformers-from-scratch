# --- Training Function ---
def train_model(model, data_loader, loss_function, optimizer, num_epochs, device='cpu'):
    model.to(device)
    model.train() # Set model to training mode
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (text_tokens, image_features, target_labels) in enumerate(data_loader):
            text_tokens, image_features, target_labels = text_tokens.to(device), image_features.to(device), target_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(text_tokens, image_features)

            # Calculate loss
            # Reshape for CrossEntropyLoss: (batch_size * combined_seq_len, output_dim)
            # and (batch_size * combined_seq_len) for labels.
            loss = loss_function(output.view(-1, output.size(-1)), target_labels.view(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training finished.")