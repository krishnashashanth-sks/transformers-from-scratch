import torch

#  Evaluation Function (`evaluate_epoch`)
def evaluate_epoch(model, dataloader, combined_loss_fn, text_gen_head, multimodal_clf_head, 
                   tool_head, text_max_seq_len, embed_dim, num_classes, device):
    model.eval()
    total_val_loss = 0
    all_individual_losses = {k: 0.0 for k in combined_loss_fn.weights.keys()}

    with torch.no_grad():
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

            # Forward pass to get multimodal output
            multimodal_output = model(text_tokens, images, audio_spectrograms, videos)

            # Calculate loss for evaluation
            current_total_loss, current_individual_losses = combined_loss_fn(
                model_output=multimodal_output,
                text_targets=text_targets,
                image_targets=image_targets,
                audio_targets=audio_targets,
                video_targets=video_targets,
                text_max_seq_len=text_max_seq_len,
                multimodal_alignment_embeddings_a=alignment_a,
                multimodal_alignment_embeddings_b=alignment_b
            )
            total_val_loss += current_total_loss.item()
            for k, v in current_individual_losses.items():
                all_individual_losses[k] += v

    avg_total_val_loss = total_val_loss / len(dataloader)
    avg_individual_losses = {k: v / len(dataloader) for k, v in all_individual_losses.items()}
    return avg_total_val_loss, avg_individual_losses
