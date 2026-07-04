import torch
from utils import decode_from_latent

def generate_sequence(model, d_latent, max_len, sos_token, eos_token, pad_token, device, latent_code=None):
    model.eval()

    with torch.no_grad():
        if latent_code is None:
            z = torch.randn(1, d_latent, device=device)
        else:
            z = latent_code.to(device)

        generated_string = decode_from_latent(model, z, max_len, sos_token, eos_token, pad_token, device)

        return generated_string