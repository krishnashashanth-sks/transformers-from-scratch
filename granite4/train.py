import time
import torch
from losses import calculate_perplexity

def train_model(num_epochs,train_dataloader,validation_dataloader,model,optimizer,lr_scheduler,vocab_size,max_grad_norm,loss_function,device):
    # Initialize variables for tracking metrics
    training_losses = []
    validation_losses = []
    training_perplexities = []
    validation_perplexities = []

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0
        train_steps = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Shift logits and labels for causal language modeling
            # The model predicts the next token, so we compare logits[..., :-1] with labels[..., 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate loss
            # Flatten the tensors for CrossEntropyLoss
            loss = loss_function(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Gradient clipping
            optimizer.step()
            lr_scheduler.step() # Update learning rate
            optimizer.zero_grad() # Reset gradients

            total_train_loss += loss.item()
            train_steps += 1

            if (step + 1) % 10 == 0: # Log every 10 steps or adjust as needed
                current_train_loss = total_train_loss / train_steps
                current_train_perplexity = calculate_perplexity(current_train_loss)
                print(f"  Step {step + 1}, Train Loss: {current_train_loss:.4f}, Train Perplexity: {current_train_perplexity:.2f}")

        avg_train_loss = total_train_loss / train_steps
        avg_train_perplexity = calculate_perplexity(avg_train_loss)
        training_losses.append(avg_train_loss)
        training_perplexities.append(avg_train_perplexity)

        print(f"Epoch {epoch + 1} finished. Avg Train Loss: {avg_train_loss:.4f}, Avg Train Perplexity: {avg_train_perplexity:.2f}")

        # Validation loop
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        val_steps = 0

        with torch.no_grad(): # No gradient calculation during validation
            for batch in validation_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask=attention_mask)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = loss_function(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / val_steps
        avg_val_perplexity = calculate_perplexity(avg_val_loss)
        validation_losses.append(avg_val_loss)
        validation_perplexities.append(avg_val_perplexity)

        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {avg_val_perplexity:.2f}")
