import math
import torch
from tqdm.auto import tqdm

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval() # Set the model to evaluation mode
    total_eval_loss = 0
    total_tokens = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(dataloader, desc="Evaluating"): # Use tqdm for progress bar
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Perform a forward pass
            outputs = model(input_ids, mask=attention_mask.unsqueeze(1).unsqueeze(2))

            # Reshape outputs and labels for CrossEntropyLoss
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = loss_fn(outputs, labels)
            total_eval_loss += loss.item() * labels.numel() # Accumulate loss weighted by number of tokens
            total_tokens += labels.numel()

    average_loss = total_eval_loss / total_tokens
    perplexity = math.exp(average_loss) # Calculate perplexity

    return average_loss, perplexity