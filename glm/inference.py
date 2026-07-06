import torch

def generate_text(model, tokenizer, prompt, max_seq_len, device):
    model.eval() # Set the model to evaluation mode
    generated_ids = []

    # Tokenize the prompt
    prompt_tokens = tokenizer.tokenize(prompt)
    prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)

    # Convert prompt IDs to a PyTorch tensor and move to device
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradient calculations
        for _ in range(max_seq_len):
            # Create attention mask for the current input_ids
            current_seq_len = input_ids.shape[1]
            # The attention mask should have 1s for actual tokens and 0s for padding, but since we are generating token by token,
            # and not padding during generation (just extending the sequence), the mask will be all ones.
            # For the MultiHeadSelfAttention module, the mask needs to be (batch_size, 1, 1, seq_len)
            attention_mask = torch.ones(1, 1, 1, current_seq_len, dtype=torch.long).to(device)

            # Get predictions from the model
            # We only care about the prediction for the last token in the sequence to predict the next token
            outputs = model(input_ids, mask=attention_mask)
            # Get the logits for the last token
            next_token_logits = outputs[0, -1, :]

            # Apply greedy decoding: select the token with the highest probability
            next_token_id = torch.argmax(next_token_logits).item()

            # Check for end-of-sequence token if defined. For simplicity, we skip this for now.
            # If tokenizer had an EOS token, we would check: if next_token_id == tokenizer.eos_id: break

            # Append the newly generated token ID to the current input sequence
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=-1)
            generated_ids.append(next_token_id)

            # Break if max_seq_len is reached (already handled by loop range)

    # Convert the entire sequence of generated token IDs back into readable text
    # We want to convert the prompt_ids + generated_ids back to text
    full_generated_ids = prompt_ids + generated_ids
    generated_text = tokenizer.convert_ids_to_tokens(full_generated_ids)
    return ' '.join(generated_text)