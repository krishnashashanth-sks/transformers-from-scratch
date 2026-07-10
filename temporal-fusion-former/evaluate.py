import torch

def evaluate_model(model, dataloader, loss_function, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch in dataloader:
            # Extract features and move to device
            static_categorical_data = [t.to(device) for t in batch['static_categorical_data']]
            static_real_data = batch['static_real_data'].to(device)
            historical_known_categorical_data = [t.to(device) for t in batch['historical_known_categorical_data']]
            historical_known_real_data = batch['historical_known_real_data'].to(device) 
            historical_unknown_categorical_data = [t.to(device) for t in batch['historical_unknown_categorical_data']]
            historical_unknown_real_data = batch['historical_unknown_real_data'].to(device) 
            future_known_categorical_data = [t.to(device) for t in batch['future_known_categorical_data']]
            future_known_real_data = batch['future_known_real_data'].to(device) 
            future_target = batch['future_target'].to(device)

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

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0
