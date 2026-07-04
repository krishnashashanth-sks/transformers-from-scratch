import torch
import torch.nn.functional as F

def vae_loss(recon_x_logits, x, mu, log_var, reduction='mean'):
    # Reconstruction Loss: Cross-Entropy between predicted logits and true target bytes
    # recon_x_logits: (batch_size, seq_len, vocab_size)
    # x: (batch_size, seq_len) - true target bytes

    recon_x_logits_flat = recon_x_logits.view(-1, recon_x_logits.size(-1))
    x_flat = x.view(-1)

    recon_loss = F.nll_loss(recon_x_logits_flat, x_flat, reduction=reduction)

    # KL Divergence Loss: D_KL(N(mu, sigma^2) || N(0, 1))
    # -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    if reduction == 'mean':
        kl_divergence /= x.size(0) # Divide by batch size

    return recon_loss, kl_divergence

