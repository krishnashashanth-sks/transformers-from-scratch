import torch

def generate_text(model, tokenizer, prompt, vocab_size,max_new_tokens=50, temperature=0.8):
    model.eval() # Set the model to evaluation mode

    # Convert prompt to input_ids
    # For this example, we'll use a simplified tokenizer or direct ID conversion
    # In a real scenario, you'd use a Hugging Face tokenizer or similar.
    # Assuming `tokenizer` can convert text to a list of token IDs
    # and `vocab_size` is from the model's setup
    if isinstance(prompt, str):
        # Dummy tokenizer: simple split and map to random IDs within vocab_size
        # This needs to be replaced with a real tokenizer if vocab_size is meaningful
        input_tokens = [hash(word) % vocab_size for word in prompt.split()]
        input_ids = torch.tensor([input_tokens], dtype=torch.long)
    else:
        input_ids = prompt # Assume prompt is already a tensor of input_ids

    generated_ids = input_ids.tolist()

    for _ in range(max_new_tokens):
        # Limit input to max_seq_len for the model
        current_input = torch.tensor(generated_ids, dtype=torch.long)[:, -model.max_seq_len:]

        with torch.no_grad():
            output = model(current_input)

        # Get the logits for the last token
        # output shape: (batch_size, seq_len, vocab_size)
        last_token_logits = output[:, -1, :]

        # Apply temperature for sampling
        if temperature > 0:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(last_token_logits, dim=-1)

        # Append the new token to the generated sequence
        generated_ids[0].append(next_token.item())

        # Stop if an EOS token is generated (dummy EOS here, assume 0 for example)
        if next_token.item() == 0: # Assuming 0 is a dummy EOS token for simplicity
            break

    # Dummy detokenizer: convert IDs back to a string (e.g., 'token_id1 token_id2')
    # In a real scenario, this would use the tokenizer's decode method.
    output_text = " ".join(map(str, generated_ids[0]))
    return output_text
