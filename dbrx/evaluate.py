import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_model(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_token_id: int
) -> tuple[float, float]:
    model.eval() # Set model to evaluation mode
    total_nll_sum = 0.0 # Accumulate sum of negative log-likelihoods for all valid tokens
    num_valid_tokens = 0 # Accumulate count of all non-padded tokens

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, batch in enumerate(val_dataloader):
            # Unpack the tuple of tensors from the DataLoader
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Perform forward pass
            outputs = model(input_ids, mask=attention_mask)

            # Reshape outputs and labels for CrossEntropyLoss
            # outputs shape: (batch_size, seq_len, vocab_size)
            # labels shape: (batch_size, seq_len)
            logits = outputs.view(-1, outputs.size(-1))
            flat_labels = labels.view(-1)

            # Calculate the batch loss. If criterion is CrossEntropyLoss with reduction='mean',
            # batch_loss.item() is the average NLL per valid token in this batch.
            batch_avg_nll = criterion(logits, flat_labels)

            non_pad_mask = (flat_labels != pad_token_id)
            current_num_valid_tokens_in_batch = non_pad_mask.sum().item()

            if current_num_valid_tokens_in_batch > 0:
                # To get the sum of NLLs for this batch, multiply the average by the count of valid tokens.
                total_nll_sum += batch_avg_nll.item() * current_num_valid_tokens_in_batch
                num_valid_tokens += current_num_valid_tokens_in_batch

    if num_valid_tokens == 0:
        avg_val_loss = 0.0
        perplexity = float('inf') # Infinite perplexity for no valid tokens
    else:
        # Calculate avg_val_loss by dividing the total sum of NLLs by the total number of valid tokens.
        avg_val_loss = total_nll_sum / num_valid_tokens
        # Perplexity is the exponential of the average negative log-likelihood.
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

    return avg_val_loss, perplexity
