import torch

def eval_step(model, input_ids, attention_mask, labels):
    """Performs a single evaluation step: forward pass, loss calculation, and accuracy computation."""
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # Disable gradient calculations
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        logits = outputs["logits"]

    # Calculate accuracy
    predictions = torch.argmax(logits, dim=-1)
    correct_predictions = (predictions == labels).sum().item()
    accuracy = correct_predictions / labels.size(0)

    return loss.item(), accuracy