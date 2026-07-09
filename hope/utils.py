import torch
import torch.nn.functional as F

def calculate_perplexity(logits, targets, ignore_index=-100):
    """
    Calculates perplexity for language modeling.

    Args:
        logits (torch.Tensor): Predicted logits of shape (batch_size, seq_len, vocab_size).
        targets (torch.Tensor): True target token IDs of shape (batch_size, seq_len).
        ignore_index (int): Index to ignore in targets (e.g., padding token).

    Returns:
        float: The calculated perplexity.
    """
    # Ensure targets are long type for CrossEntropyLoss
    targets = targets.long()

    # CrossEntropyLoss expects (N, C, ...) for input and (N, ...) for target
    # Reshape logits to (batch_size * seq_len, vocab_size)
    # Reshape targets to (batch_size * seq_len)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)

    # Perplexity is exp(loss)
    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_accuracy(logits, targets, ignore_index=-100):
    """
    Calculates token prediction accuracy.

    Args:
        logits (torch.Tensor): Predicted logits of shape (batch_size, seq_len, vocab_size).
        targets (torch.Tensor): True target token IDs of shape (batch_size, seq_len).
        ignore_index (int): Index to ignore in targets (e.g., padding token).

    Returns:
        float: The calculated accuracy (0-1 range).
    """
    # Get the predicted token IDs
    predictions = torch.argmax(logits, dim=-1)

    # Create a mask to ignore specific indices (e.g., padding)
    mask = (targets != ignore_index)

    # Calculate correct predictions only where mask is True
    correct_predictions = (predictions == targets) & mask

    # Sum up correct predictions and divide by the total number of non-ignored tokens
    accuracy = correct_predictions.sum().item() / mask.sum().item()
    return accuracy

