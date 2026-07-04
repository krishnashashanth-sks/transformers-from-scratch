import torch

# Helper function to generate masks
def create_src_mask(src, pad_token):
    # src has shape (batch_size, seq_len)
    # Mask where src is not the padding token (True for non-padding, False for padding)
    # Reshape to (batch_size, 1, 1, seq_len) for broadcasting with attention scores
    return (src != pad_token).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt, pad_token):
    # tgt has shape (batch_size, seq_len)

    # 1. Target Padding Mask (similar to src_mask)
    # Mask where tgt is not the padding token
    # Shape: (batch_size, 1, 1, seq_len)
    tgt_pad_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)

    # 2. Causal (Look-ahead) Mask
    # Prevents attention to future tokens
    # Shape: (seq_len, seq_len)
    seq_len = tgt.size(1)
    tgt_causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device))
    # Reshape to (1, 1, seq_len, seq_len) to broadcast with batch_size
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(0)

    # 3. Combine both masks using logical AND
    final_tgt_mask = tgt_pad_mask & tgt_causal_mask

    return final_tgt_mask

def preprocess_sequence(sequence, max_len, sos_token, eos_token, pad_token):
    # Convert string to byte values (integers 0-255)
    byte_values = list(sequence.encode('utf-8'))

    # Prepend SOS and append EOS
    processed_sequence = [sos_token] + byte_values + [eos_token]

    # Handle truncation or padding
    if len(processed_sequence) > max_len:
        # Truncate: Keep SOS, remove EOS, truncate the middle part, then re-add EOS
        if max_len < 2: # Cannot even fit SOS and EOS
            processed_sequence = [sos_token] * max_len
        elif max_len == 2:
            processed_sequence = [sos_token, eos_token]
        else:
            processed_sequence = [sos_token] + processed_sequence[1:max_len - 1] + [eos_token]
    elif len(processed_sequence) < max_len:
        # Pad: Add PAD_TOKENs to the end
        processed_sequence = processed_sequence + [pad_token] * (max_len - len(processed_sequence))

    processed_sequence = processed_sequence[:max_len] # Ensure final length is max_len

    return torch.LongTensor(processed_sequence)

def decode_from_latent(model, latent_code, max_len, sos_token, eos_token, pad_token, device):
    model.eval()

    with torch.no_grad():
        decoder_input = torch.tensor([[sos_token]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = create_tgt_mask(decoder_input, pad_token)

            decoder_output_features = model.decoder(decoder_input, latent_code, tgt_mask)

            output_logits = model.output_head(decoder_output_features)

            last_token_logits = output_logits[:, -1, :]

            next_token = last_token_logits.argmax(dim=-1, keepdim=True)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if next_token.item() == eos_token:
                break

        generated_sequence_ids = decoder_input.squeeze(0).tolist()

        try:
            eos_pos = generated_sequence_ids.index(eos_token)
            generated_sequence_ids = generated_sequence_ids[:eos_pos]
        except ValueError:
            pass

        if generated_sequence_ids and generated_sequence_ids[0] == sos_token:
            generated_sequence_ids = generated_sequence_ids[1:]

        byte_list = [token_id for token_id in generated_sequence_ids if token_id >= 3] # Assuming special tokens are < 3
        try:
            decoded_string = bytes(byte_list).decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            decoded_string = "<UnicodeDecodeError>"

        return decoded_string

def reconstruct_sequence(model, input_sequence_tensor, max_len, sos_token, eos_token, pad_token, device):
    model.eval()

    with torch.no_grad():
        encoder_input = input_sequence_tensor.to(device).unsqueeze(0)

        src_mask = create_src_mask(encoder_input, pad_token)

        encoder_output = model.encoder(encoder_input, src_mask)

        z, mu, log_var = model.latent_module(encoder_output)

        reconstructed_string = decode_from_latent(model, z, max_len, sos_token, eos_token, pad_token, device)

        return reconstructed_string