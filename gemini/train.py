import torch
from utils import check_nan_in_tensor

#  Training Function (`train_epoch`)
def train_epoch(model, dataloader, combined_loss_fn, optimizer, scheduler, monitor, 
                text_gen_head, multimodal_clf_head, tool_head, text_max_seq_len, 
                grad_clip_norm, device):
    model.train()
    total_epoch_loss = 0
    for batch_idx, batch_data in enumerate(dataloader):
        # Move data to device
        text_tokens = batch_data['text_tokens'].to(device)
        images = batch_data['images'].to(device)
        audio_spectrograms = batch_data['audio_spectrograms'].to(device)
        videos = batch_data['videos'].to(device)

        text_targets = batch_data['text_targets'].to(device)
        image_targets = batch_data['image_targets'].to(device)
        audio_targets = batch_data['audio_targets'].to(device)
        video_targets = batch_data['video_targets'].to(device)
        alignment_a = batch_data['multimodal_alignment_embeddings_a'].to(device)
        alignment_b = batch_data['multimodal_alignment_embeddings_b'].to(device)

        optimizer.zero_grad()

        # Forward pass
        multimodal_output = model(text_tokens, images, audio_spectrograms, videos)

        # Calculate loss
        total_loss, individual_losses = combined_loss_fn(
            model_output=multimodal_output,
            text_targets=text_targets,
            image_targets=image_targets,
            audio_targets=audio_targets,
            video_targets=video_targets,
            text_max_seq_len=text_max_seq_len,
            multimodal_alignment_embeddings_a=alignment_a,
            multimodal_alignment_embeddings_b=alignment_b
        )
        
        # NaN detection
        if check_nan_in_tensor(total_loss, name='total_loss'):
            print("NaN loss detected during training. Skipping backward pass for this batch.")
            continue

        # Backward pass and optimization
        total_loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step() # Step LR scheduler per batch

        total_epoch_loss += total_loss.item()

        # Logging
        monitor.log_losses(total_loss, individual_losses, device)
        monitor.log_gradient_norms(model, grad_clip_norm)
        monitor.log_resource_utilization()
        monitor.step()

    return total_epoch_loss / len(dataloader)
