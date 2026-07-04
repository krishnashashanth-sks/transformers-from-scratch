import torch
import torch.nn.functional as F
from main import SEQ_LEN

def generate_text(model, tokenizer, prompt, max_length=100, device='cpu', temperature=1.0, top_k=0):
    model.eval() # Set model to evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # Get the model's prediction for the next token
            # We only need the prediction for the last token in the sequence
            output = model(generated_ids)
            next_token_logits = output[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if the model predicts the end-of-sequence token (if tokenizer has one)
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

            # Optional: Break if sequence length exceeds SEQ_LEN, though this is for generation
            # and might need adjustment if model's `seq_len` is a strict limit.
            # For this model, `seq_len` is used in init but not enforced during inference in a strict sliding window manner.
            # We will rely on max_length and eos_token for termination.
            if generated_ids.shape[1] >= SEQ_LEN: # Limit to model's max sequence length if necessary
                break

    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text