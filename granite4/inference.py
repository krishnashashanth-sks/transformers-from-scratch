import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length, temperature=1.0, top_k=None, top_p=None, device='cpu'):
    model.eval() # Set model to evaluation mode

    # 1. Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Initialize generated sequence with prompt tokens
    generated_ids = input_ids

    # 2. Implement generation loop
    for _ in range(max_length - input_ids.shape[1]): # Generate up to max_length tokens
        with torch.no_grad():
            # Get logits for the next token
            # Model expects full sequence, but we only care about the last token's prediction
            output_logits = model(generated_ids)
            next_token_logits = output_logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply Top-K sampling
            if top_k is not None:
                # Set to -infinity all but the top_k values
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply Top-P (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold (token will still be selected if it's the first)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep at least one token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('Inf')

            # Convert logits to probabilities and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append the sampled token ID
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # 3. Decode the entire sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text
