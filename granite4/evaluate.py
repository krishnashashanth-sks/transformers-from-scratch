import torch

def evaluate_model(model,test_dataloader,loss_function,vocab_size,device):
    model.eval() # Set model to evaluation mode
    total_test_loss = 0
    test_steps = 0

    with torch.no_grad(): # No gradient calculation during evaluation
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate loss
            loss = loss_function(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            total_test_loss += loss.item()
            test_steps += 1
    return total_test_loss,test_steps