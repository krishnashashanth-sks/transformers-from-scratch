import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from evaluate import evaluate_model

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler, # Or a custom scheduler
    scaler: GradScaler, # For mixed precision
    device: torch.device,
    epochs: int,
    accumulation_steps: int,
    log_interval: int,
    checkpoint_dir: str, # Directory to save checkpoints
    pad_token_id: int
):
    model.to(device)
    global_step = 0
    best_val_loss = float('inf')

    # Make sure checkpoint directory exists
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Unpack the tuple of tensors from the DataLoader
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device) # Target labels for LM task

            with autocast(): # Enables mixed precision
                outputs = model(input_ids, mask=attention_mask)

                # Reshape outputs and labels for CrossEntropyLoss
                # outputs shape: (batch_size, seq_len, vocab_size)
                # labels shape: (batch_size, seq_len)
                logits = outputs.view(-1, outputs.size(-1))
                flat_labels = labels.view(-1)
                loss = criterion(logits, flat_labels)
                loss = loss / accumulation_steps # Scale loss for gradient accumulation

            scaler.scale(loss).backward() # Scale loss and perform backward pass
            total_loss += loss.item() * accumulation_steps # Unscale for logging

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                # Unscale gradients, clip, and update weights
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Apply clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step() # Update learning rate
                global_step += 1

            if global_step % log_interval == 0 and (batch_idx + 1) % accumulation_steps == 0:
                avg_batch_loss = total_loss / (log_interval / accumulation_steps)
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch}, Step: {global_step}, Train Loss: {avg_batch_loss:.4f}, LR: {current_lr:.6f}")
                total_loss = 0 # Reset total_loss for the next log_interval

        # --- Validation Loop ---
        # Call the standalone evaluate_model function
        avg_val_loss, perplexity = evaluate_model(
            model=model,
            val_dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            pad_token_id=pad_token_id
        )
        print(f"Epoch: {epoch}, Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")

        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'scaler_state_dict': scaler.state_dict() # Save scaler state
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path} with Val Loss: {best_val_loss:.4f}")

    print("Training complete.")
