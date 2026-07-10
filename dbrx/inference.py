import torch
import torch.nn as nn

def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    num_beams: int = 1 # Conceptual guidance for beam search will be provided later
) -> str:
    model.eval() # Set model to evaluation mode

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # List to store generated tokens
    generated_ids = input_ids.tolist()[0]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Prepare input for the model
            # If your model's forward expects (batch_size, seq_len) and handles attention mask internally,
            # you might need to pass the attention mask or ensure padding is handled.
            # For simplicity, we'll just pass the current sequence.
            current_input = torch.tensor(generated_ids).unsqueeze(0).to(device)

            # Get logits for the next token
            # We only care about the logits for the last token in the sequence
            outputs = model(current_input)
            next_token_logits = outputs[:, -1, :]

            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Decoding strategy: sampling or greedy
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    # Set all but the top_k values to -inf
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from the probability distribution
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding: select the token with the highest probability
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Append the generated token
            generated_ids.append(next_token.item())

            # Stop if end-of-sequence token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated token IDs back into text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text
