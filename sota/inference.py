import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, device='cpu'):
    model.eval() # Set model to evaluation mode

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions for the current sequence
            outputs = model(input_ids)
            # We only care about the last token's logits for the next prediction
            next_token_logits = outputs[0, -1, :] / temperature

            # Apply softmax to get probabilities
            probabilities = F.softmax(next_token_logits, dim=-1)

            # Sample from the distribution
            next_token_id = torch.multinomial(probabilities, num_samples=1)

            # If it's the EOS token, stop generation
            if next_token_id == tokenizer.eos_token_id:
                break

            # Append the predicted token to the generated sequence
            generated_ids.append(next_token_id.item())

            # Add the new token to the input_ids for the next iteration
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text