import torch

def generate_sequence(model, tokenizer, prompt_ids, max_new_tokens, device='cpu'):
    model.eval() # Set the model to evaluation mode
    input_ids = prompt_ids.to(device) # Start with the prompt
    generated_sequence = input_ids.tolist()[0] # Convert to list for easy appending

    print(f"Generating from prompt: {tokenizer.decode(generated_sequence)}")

    for _ in range(max_new_tokens):
        # The model's forward pass generates fast weights dynamically
        # for the current input_ids and produces logits for the next token.
        # Only consider the last token for prediction if input_ids > model.embedding_layer.max_len
        current_input_ids = input_ids if input_ids.size(1) <= model.embedding_layer.max_len else input_ids[:, -model.embedding_layer.max_len:]

        with torch.no_grad():
            output_logits, _ = model(current_input_ids)

        # Get logits for the last token in the sequence
        next_token_logits = output_logits[:, -1, :]

        # Sample the next token (e.g., using argmax for greedy decoding)
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) # unsqueeze for batch dim

        # Append the new token to the sequence
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        generated_sequence.append(next_token_id.item())

        # Print token by token generation
        print(tokenizer.decode([next_token_id.item()]), end=' ')

    print(f"\n\nGenerated sequence: {tokenizer.decode(generated_sequence)}")
    return generated_sequence
