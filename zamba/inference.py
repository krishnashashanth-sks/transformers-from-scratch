import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=None, device='cpu'):
    model.eval() # Set the model to evaluation mode
    encoded_input = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated_sequence = encoded_input

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated_sequence)
            # Get logits for the last token
            logits = outputs[:, -1, :] / temperature

            if top_k is not None:
                # Apply top-k sampling
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                probabilities = F.softmax(top_k_values, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)
                next_token_id = top_k_indices.gather(-1, next_token_id)
            else:
                # Sample from the full distribution
                probabilities = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)

            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat([generated_sequence, next_token_id], dim=-1)

            # Stop if the model generates the EOS token (if defined and desired)
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break

    decoded_output = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    return decoded_output
