def train_model(num_epochs,model,train_dataloader,optimizer,loss_function,device):
    print(f"Starting training for {num_epochs} epoch(s)...")

    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        total_loss = 0.0
        num_batches = 0

        # Iterate through each batch in the train_dataloader
        for i, batch in enumerate(train_dataloader):
            # Extract features and move to device
            static_categorical_data = [t.to(device) for t in batch['static_categorical_data']]
            static_real_data = batch['static_real_data'].to(device)
            historical_known_categorical_data = [t.to(device) for t in batch['historical_known_categorical_data']]
            historical_known_real_data = batch['historical_known_real_data'].to(device) # FIX: pass as tensor, not list
            historical_unknown_categorical_data = [t.to(device) for t in batch['historical_unknown_categorical_data']]
            historical_unknown_real_data = batch['historical_unknown_real_data'].to(device) # FIX: pass as tensor, not list
            future_known_categorical_data = [t.to(device) for t in batch['future_known_categorical_data']]
            future_known_real_data = batch['future_known_real_data'].to(device) # FIX: pass as tensor, not list
            future_target = batch['future_target'].to(device)

            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(
                static_categorical_data,
                static_real_data,
                historical_known_categorical_data,
                historical_known_real_data,
                historical_unknown_categorical_data,
                historical_unknown_real_data,
                future_known_categorical_data,
                future_known_real_data
            )

            # Calculate loss
            loss = loss_function(predictions, future_target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

    print("Training complete.")