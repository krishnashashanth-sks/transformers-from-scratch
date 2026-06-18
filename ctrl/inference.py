import torch

# --- 11. Model Inference Function ---

def generate_text(model, control_code_id, prompt_ids, max_new_tokens, device, eos_token_id=None):
    """
    Generates text using the trained CTRL-like model.

    Args:
        model (nn.Module): The BasicCTRLModel instance.
        control_code_id (torch.Tensor): Tensor containing the token ID for the control code (shape: [1, 1]).
        prompt_ids (torch.Tensor): Tensor containing the token IDs for the initial prompt (shape: [1, seq_len]).
        max_new_tokens (int): Maximum number of new tokens to generate.
        device (torch.device): The device (cpu/cuda) to run the inference on.
        eos_token_id (int, optional): The End-Of-Sequence token ID. Generation stops if this token is predicted.

    Returns:
        list[int]: A list of generated token IDs, including control code and prompt.
    """
    model.eval() # Set model to evaluation mode
    generated_sequence_ids = torch.cat((control_code_id, prompt_ids), dim=1).to(device)

    with torch.no_grad(): # Disable gradient calculations
        for _ in range(max_new_tokens):
            # The model's input_ids should not exceed MAX_SEQ_LEN
            # If current sequence is too long, take the last MAX_SEQ_LEN tokens
            current_input_ids = generated_sequence_ids
            if current_input_ids.size(1) > MAX_SEQ_LEN:
                current_input_ids = current_input_ids[:, -MAX_SEQ_LEN:]

            # Create attention mask for the current input
            attention_mask = torch.ones_like(current_input_ids, dtype=torch.long).to(device)

            # Get model predictions
            outputs = model(current_input_ids, attention_mask=attention_mask)

            # Get the logits for the last token in the sequence
            next_token_logits = outputs[:, -1, :] # Shape: [batch_size, vocab_size]

            # Sample the next token (using argmax for greedy decoding for simplicity)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) # Shape: [1, 1]

            # Append the new token to the generated sequence
            generated_sequence_ids = torch.cat((generated_sequence_ids, next_token_id), dim=1)

            # Check for EOS token
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

    return generated_sequence_ids.squeeze(0).tolist() # Return as a list of IDs