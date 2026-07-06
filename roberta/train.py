
import torch

def train_step(model, optimizer, input_ids, attention_mask, labels):
    """Performs a single training step: forward pass, loss calculation, backpropagation, and parameter update."""
    model.train() # Set the model to training mode

    optimizer.zero_grad() # Zero the gradients

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs["loss"]
    logits = outputs["logits"]

    # Backward pass
    loss.backward()

    # Update model parameters
    optimizer.step()

    return loss.item(), logits