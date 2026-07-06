import torch

def predict(model, input_ids, attention_mask):
    """Performs inference: forward pass and returns predicted labels."""
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # Disable gradient calculations
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs["logits"]

    # Get predicted labels
    predicted_labels = torch.argmax(logits, dim=-1)

    return predicted_labels