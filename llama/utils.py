def tokenize(text):
    # Convert to lowercase and split into words
    words = []
    temp_words = text.lower().replace('.', '').split()
    for word in temp_words:
        if word == '[mask]': # Recognize the special token string
            words.append('[MASK]')
        else:
            words.append(word)

    # Convert words to token IDs, handle unknown words
    token_ids = []
    for word in words:
        token_ids.append(vocab.get(word, vocab['[UNK]']))
    return token_ids

max_seq_len_llama = 50 # Re-using max_seq_len from Llama model definition for consistency

def prepare_llama_causal_lm_input(text, max_seq_len, pad_id):
    # Tokenize the input text
    token_ids = tokenize(text)

    # For causal language modeling, input_ids are the tokens, and labels are the next tokens.
    # So, labels are essentially input_ids shifted by one position.

    # Truncate if sequence is too long
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    # Prepare input_ids and labels
    input_ids = token_ids
    # Labels are shifted by one. The last token's label is pad_id (or could be original_token_id)
    # For simplicity, we assume we want to predict the next token, so labels are the original tokens shifted.
    # The first token's label is effectively the second token, etc.
    # The model predicts token[i+1] given token[i].
    # Here, for a sequence of length L, we have L inputs and L labels. Labels[i] is the target for input_ids[i].
    # If input_ids = [T1, T2, T3], then labels = [T2, T3, pad_id] assuming we predict next token.
    # A common convention is that labels are the input_ids themselves, but the loss is only computed for positions
    # where the model is meant to predict. So, for input [T1, T2, T3], labels are [T1, T2, T3].
    # The attention mask ensures the model only sees previous tokens for prediction at each step.
    # The loss function then aligns logits at position `i` with `labels` at `i` (which is `input_ids[i]`).
    # We'll use the common convention: labels are input_ids.
    labels = token_ids # For causal LM, labels are the input tokens themselves, with a causal mask.

    # Pad sequences to max_seq_len
    padding_len = max_seq_len - len(input_ids)
    input_ids_padded = input_ids + [pad_id] * padding_len
    labels_padded = labels + [pad_id] * padding_len # Labels are also padded

    # Create causal attention mask
    # A causal mask is triangular, allowing attention only to previous tokens.
    # For Llama, the attention_mask is typically created within the model's forward pass
    # based on the input length, but we can generate a basic one here for the dataset if needed.
    # For `LlamaForCausalLM` forward, the `attention_mask` passed is usually for padding.
    # The causal mask is implicitly generated in `MultiHeadAttention` via `masked_fill`.
    # So, for the dataset, we primarily need a padding mask.

    # This attention_mask will be a padding mask, indicating which tokens are real (1) and which are padding (0).
    padding_attention_mask = [1] * len(input_ids) + [0] * padding_len

    return input_ids_padded, labels_padded, padding_attention_mask