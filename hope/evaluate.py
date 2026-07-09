import torch
import math
import torch.nn.functional as F

def evaluate_model(model, data_loader, vocab_size, device='cpu', ignore_index=-100):
    model.eval() # Set the model to evaluation mode
    total_eval_loss = 0
    total_eval_correct = 0
    total_eval_tokens = 0
    eval_batches = 0

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch_idx, (input_ids_batch, target_ids_batch) in enumerate(data_loader):
            input_ids_batch = input_ids_batch.to(device)
            target_ids_batch = target_ids_batch.to(device)

            output_logits, _ = model(input_ids_batch)

            # Calculate loss for perplexity
            loss_input = output_logits.view(-1, vocab_size)
            loss_target = target_ids_batch.view(-1)

            loss = F.cross_entropy(loss_input, loss_target, ignore_index=ignore_index)
            total_eval_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(output_logits, dim=-1)
            mask = (target_ids_batch != ignore_index)
            total_eval_correct += (predictions == target_ids_batch)[mask].sum().item()
            total_eval_tokens += mask.sum().item()
            eval_batches += 1

    avg_eval_loss = total_eval_loss / eval_batches
    avg_eval_perplexity = math.exp(avg_eval_loss)
    avg_eval_accuracy = total_eval_correct / total_eval_tokens if total_eval_tokens > 0 else 0

    print(f"\n--- Evaluation Results ---")
    print(f"Average Loss: {avg_eval_loss:.4f}")
    print(f"Perplexity: {avg_eval_perplexity:.2f}")
    print(f"Average Accuracy: {avg_eval_accuracy:.4f}")
    print(f"--------------------------")

    return avg_eval_loss, avg_eval_perplexity, avg_eval_accuracy