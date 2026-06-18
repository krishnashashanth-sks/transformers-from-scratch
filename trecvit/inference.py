import torch

def predict_class_label(model, video_tensor, idx_to_class, device):
    model.eval() # Set model to evaluation mode

    # Move input tensor to the correct device
    video_tensor = video_tensor.to(device)

    with torch.no_grad(): # Disable gradient calculation for inference
        # Add a batch dimension if it's a single video (C, T, H, W) -> (1, C, T, H, W)
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)

        outputs = model(video_tensor) # Forward pass
        probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
        predicted_class_id = torch.argmax(probabilities, dim=1).item() # Get the class with highest probability

    predicted_label = idx_to_class.get(predicted_class_id, f"Unknown class ID: {predicted_class_id}")
    return predicted_label, probabilities[0, predicted_class_id].item() # Return label and its probability