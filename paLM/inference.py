import torch

# --- Inference Function ---
def infer_model(model, data_loader, device='cpu'):
    model.to(device)
    model.eval() # Set model to evaluation mode

    all_predictions = []
    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, (text_tokens, image_features, _) in enumerate(data_loader):
            text_tokens, image_features = text_tokens.to(device), image_features.to(device)

            output = model(text_tokens, image_features)
            
            # For demonstration, let's get predicted token indices
            predicted_token_indices = torch.argmax(output, dim=-1)
            all_predictions.append(predicted_token_indices.cpu())

    return torch.cat(all_predictions, dim=0)