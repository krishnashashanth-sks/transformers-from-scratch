import torch

def predict_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []

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

            # Forward pass to get predictions
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

            all_predictions.append(predictions.cpu())
            all_targets.append(future_target.cpu())

    return torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0)
