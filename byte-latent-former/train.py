from tqdm.auto import tqdm
from utils import create_src_mask,create_tgt_mask
from losses import vae_loss
import torch

def train_step(model, optimizer, data_batch, pad_token, device, kl_weight):
    model.train()

    encoder_input, decoder_input, decoder_target = data_batch
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    decoder_target = decoder_target.to(device)

    optimizer.zero_grad()

    src_mask = create_src_mask(encoder_input, pad_token)
    tgt_mask = create_tgt_mask(decoder_input, pad_token)

    output_logits, mu, log_var = model(encoder_input, decoder_input, src_mask, tgt_mask)

    recon_loss, kl_divergence = vae_loss(output_logits, decoder_target, mu, log_var)

    total_loss = recon_loss + kl_weight * kl_divergence

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), recon_loss.item(), kl_divergence.item()

def evaluate_step(model, data_batch, pad_token, device, kl_weight):
    model.eval()

    with torch.no_grad():
        encoder_input, decoder_input, decoder_target = data_batch
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_target = decoder_target.to(device)

        src_mask = create_src_mask(encoder_input, pad_token)
        tgt_mask = create_tgt_mask(decoder_input, pad_token)

        output_logits, mu, log_var = model(encoder_input, decoder_input, src_mask, tgt_mask)

        recon_loss, kl_divergence = vae_loss(output_logits, decoder_target, mu, log_var)

        total_loss = recon_loss + kl_weight * kl_divergence

    return total_loss.item(), recon_loss.item(), kl_divergence.item()

def train_loop(model, train_dataloader, val_dataloader, optimizer, pad_token, device, epochs, kl_weight, log_interval=10):
    model.to(device)
    history = {'train_loss': [], 'train_recon_loss': [], 'train_kl_loss': [],
               'val_loss': [], 'val_recon_loss': [], 'val_kl_loss': []}

    print("Starting training loop...")

    for epoch in range(epochs):
        # Training Phase
        total_train_loss = 0
        total_train_recon_loss = 0
        total_train_kl_loss = 0
        train_batches = 0

        if train_dataloader: # Check if train_dataloader is not empty
            for batch_idx, data_batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")):
                train_loss, train_recon_loss, train_kl_loss = train_step(model, optimizer, data_batch, pad_token, device, kl_weight)
                total_train_loss += train_loss
                total_train_recon_loss += train_recon_loss
                total_train_kl_loss += train_kl_loss
                train_batches += 1

                if batch_idx % log_interval == 0 and batch_idx > 0:
                    avg_batch_loss = total_train_loss / train_batches
                    avg_batch_recon = total_train_recon_loss / train_batches
                    avg_batch_kl = total_train_kl_loss / train_batches
                    print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_dataloader)} | Train Loss: {avg_batch_loss:.4f}, Recon Loss: {avg_batch_recon:.4f}, KL Loss: {avg_batch_kl:.4f}")

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        avg_train_recon_loss = total_train_recon_loss / train_batches if train_batches > 0 else 0
        avg_train_kl_loss = total_train_kl_loss / train_batches if train_batches > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['train_recon_loss'].append(avg_train_recon_loss)
        history['train_kl_loss'].append(avg_train_kl_loss)

        # Validation Phase
        total_val_loss = 0
        total_val_recon_loss = 0
        total_val_kl_loss = 0
        val_batches = 0

        if val_dataloader: # Check if val_dataloader is not empty
            for data_batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                val_loss, val_recon_loss, val_kl_loss = evaluate_step(model, data_batch, pad_token, device, kl_weight)
                total_val_loss += val_loss
                total_val_recon_loss += val_recon_loss
                total_val_kl_loss += val_kl_loss
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        avg_val_recon_loss = total_val_recon_loss / val_batches if val_batches > 0 else 0
        avg_val_kl_loss = total_val_kl_loss / val_batches if val_batches > 0 else 0
        history['val_loss'].append(avg_val_loss)
        history['val_recon_loss'].append(avg_val_recon_loss)
        history['val_kl_loss'].append(avg_val_kl_loss)

        print(f"\nEpoch {epoch+1} Complete | Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        print(f"Avg Train Recon: {avg_train_recon_loss:.4f}, Avg Train KL: {avg_train_kl_loss:.4f}")
        print(f"Avg Val Recon: {avg_val_recon_loss:.4f}, Avg Val KL: {avg_val_kl_loss:.4f}\n")

    print("Training complete.")
    return history
