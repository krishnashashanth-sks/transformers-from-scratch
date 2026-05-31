import torch.nn.functional as F

def generate_text(gpt_model, start_string, num_generate, temperature, char_to_idx, idx_to_char, max_seq_len, device):
    # 2. Set the gpt_model to evaluation mode and use torch.no_grad()
    gpt_model.eval()
    with torch.no_grad():
        # 3. Convert the start_string into a numerical input tensor
        # For generation, the input sequence should ideally be max_seq_len long for consistent behavior
        # We'll use a sliding window approach, so the initial input is just the start_string
        input_chars = [char_to_idx[char] for char in start_string]
        # Pad the initial input to max_seq_len if it's shorter, or truncate if longer
        if len(input_chars) < max_seq_len:
            current_input = input_chars + [0] * (max_seq_len - len(input_chars)) # Pad with a default index (e.g., 0 for empty)
        else:
            current_input = input_chars[-max_seq_len:] # Use the last max_seq_len characters

        input_tensor = torch.tensor(current_input, dtype=torch.long, device=device).unsqueeze(0)

        # 4. Initialize an empty string to store the generated text
        generated_text = start_string

        # 5. Create a causal mask for the attention mechanism
        # The mask prevents attending to future tokens. It's applied inside the TransformerBlock.
        # For generation, we don't need to pass it explicitly to the model's forward pass
        # if the model already handles causal masking internally based on its decoder-only structure.
        # However, if the TransformerBlock expects a mask, we'll create one.
        # For this SimpleGPT, the mask is generated once for the full sequence length in the forward pass.
        # Let's recreate it here for clarity if needed.
        c_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        causal_mask = c_mask.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, seq_len, seq_len)

        # 6. Implement a loop that runs for num_generate iterations
        for _ in range(num_generate):
            # a. Take the most recent max_seq_len tokens from the current input sequence
            # This is already handled by input_tensor containing the most recent context
            # The input to the model should always be of length max_seq_len
            # So, we pass the current_input_tensor which represents the 'context'

            # b. Pass this input sequence through the gpt_model to get output_logits
            # We use the causal_mask for the forward pass
            output_logits = gpt_model(input_tensor, mask=causal_mask)

            # c. Focus on the logits for the last token in the sequence (output_logits[:, -1, :])
            last_token_logits = output_logits[:, -1, :]

            # d. Apply the temperature to the logits to control randomness
            if temperature == 0:
                # Deterministic sampling (greedy)
                predicted_idx = torch.argmax(last_token_logits, dim=-1).item()
            else:
                # Apply temperature and sample from a probability distribution
                probs = F.softmax(last_token_logits / temperature, dim=-1)
                predicted_idx = torch.multinomial(probs, num_samples=1).item()

            # e. Convert the sampled index back to a character
            predicted_char = idx_to_char[predicted_idx]
            generated_text += predicted_char

            # f. Append the generated character to the current input sequence
            # Update input_tensor for the next iteration by sliding the window
            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_idx]], dtype=torch.long, device=device)), dim=1)

    gpt_model.train() # Set model back to training mode
    return generated_text
