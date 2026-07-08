import torch
import math

def evaluate_model(model, dataloader, criterion, vocab_size, device, dummy_tokenizer):
    model.eval()
    total_eval_loss = 0
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (images, text_tokens) in enumerate(dataloader):
            images = images.to(device)
            text_tokens = text_tokens.to(device)

            logits = model(images, text_tokens)

            targets = text_tokens[:, 1:].contiguous().view(-1)
            predictions = logits[:, :-1, :].contiguous().view(-1, vocab_size)

            loss = criterion(predictions, targets)
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(dataloader)
    perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 300 else float('inf')

    print(f"Evaluation finished. Average Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")

