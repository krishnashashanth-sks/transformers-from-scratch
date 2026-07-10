import torch
import os

#  Implement NaN Detection and Health Checks
def check_nan_in_tensor(tensor, name='tensor'):
    """Checks if a tensor contains NaN values and prints a warning."""
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}!")
        return True
    return False

#  Implement Intermediate Checkpointing
def save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, filepath):
    """Saves the current training state to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath} at epoch {epoch}, step {step} with validation loss {val_loss:.4f}")
