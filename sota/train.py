import math
import torch

def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, num_epochs, device, tokenizer):
    train_losses = []
    val_losses = []
    val_perplexities = []

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_train_loss = 0
        train_steps = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-------------------- Training --------------------")

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad() # Reset optimizer's gradients

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # For language modeling, targets are the input_ids shifted by one position
            # The model predicts the next token. So, for input_ids[i], target is input_ids[i+1]
            # Example: input [A, B, C], model predicts [B, C, D]
            # Labels for current sequence are [B, C, D] (shifted input_ids)
            targets = input_ids.clone()

            # Flatten the logits and targets for CrossEntropyLoss
            # Exclude the last token from input for prediction
            # Exclude the first token from targets as it's not predicted
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))

            loss.backward() # Backward pass
            optimizer.step() # Update model parameters

            total_train_loss += loss.item()
            train_steps += 1

            if (batch_idx + 1) % 100 == 0: # Print updates periodically
                print(f"  Batch {batch_idx + 1}/{len(train_dataloader)} Loss: {loss.item():.4f} ",
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # -------------------- Validation --------------------
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        val_steps = 0

        print("-------------------- Validation --------------------")
        with torch.no_grad(): # Disable gradient calculations
            for batch_idx, batch in enumerate(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                targets = input_ids.clone()

                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = targets[..., 1:].contiguous()

                val_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))

                total_val_loss += val_loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / val_steps
        val_losses.append(avg_val_loss)

        # Calculate perplexity
        perplexity = math.exp(avg_val_loss) if avg_val_loss < 100 else float('inf') # Avoid overflow for exp
        val_perplexities.append(perplexity)

        print(f"Validation Loss: {avg_val_loss:.4f} Perplexity: {perplexity:.2f}")

        scheduler.step() # Update the learning rate scheduler

        model.train() # Set model back to training mode for next epoch

    print("\nTraining complete!")
    return train_losses, val_losses, val_perplexities
