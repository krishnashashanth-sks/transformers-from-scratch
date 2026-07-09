import torch
from tqdm.auto import tqdm

def evaluate_epoch(model, dataloader, loss_fn, device):
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Evaluation")

    with torch.no_grad(): # Disable gradient calculation during evaluation
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids)

            # Reshape outputs and labels for CrossEntropyLoss
            reshaped_outputs = outputs.view(-1, outputs.size(-1))
            reshaped_labels = labels.view(-1)

            # Calculate loss
            loss = loss_fn(reshaped_outputs, reshaped_labels)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)
