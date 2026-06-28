import torch

def generate_llama_text(model, tokenizer, idx_to_token, vocab, prompt, max_gen_len, device):
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # Disable gradient computation
        # Tokenize the prompt
        input_ids_list = tokenizer(prompt)

        # Convert to tensor, add batch dimension, and move to device
        current_input_ids = torch.LongTensor(input_ids_list).unsqueeze(0).to(device)

        # Retrieve special token IDs
        pad_id = vocab['[PAD]']
        sep_id = vocab['[SEP]'] # Assuming SEP is used to indicate end of sequence in generation
        cls_id = vocab['[CLS]'] # Also filter CLS if it appears
        mask_id = vocab['[MASK]'] # Also filter MASK if it appears
        unk_id = vocab['[UNK]'] # Also filter UNK if it appears

        # Initialize generated_token_ids with prompt tokens
        generated_token_ids = current_input_ids.tolist()[0]

        # Create a temporary extended idx_to_token for generation, handling potential out-of-vocab predictions
        extended_idx_to_token = idx_to_token.copy()
        for i in range(len(idx_to_token), model.lm_head.out_features): # vocab_size_llama is model.lm_head.out_features
            extended_idx_to_token[i] = '[UNK]' # Map unknown predicted IDs to [UNK]

        # Start iterative generation
        for _ in range(max_gen_len):
            # Create a simple padding attention_mask (all ones for the current sequence length)
            seq_len = current_input_ids.shape[1]
            attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
            # The Llama model expects the attention mask to be (batch_size, seq_len),
            # but its internal MultiHeadAttention expects (batch_size, 1, seq_len, seq_len).
            # The Llama model's forward method handles the conversion from (batch_size, seq_len)
            # to the expanded form internally for padding_attention_mask.

            # Pass current_input_ids and attention_mask to the model
            logits = model(current_input_ids, attention_mask=attention_mask)

            # Extract logits for the last token and get the next token ID (greedy decoding)
            last_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(last_token_logits, dim=-1).item()

            # If end-of-sequence token is generated, break
            if next_token_id == sep_id:
                break

            # Append next_token_id to the generated list
            generated_token_ids.append(next_token_id)

            # Update current_input_ids for the next iteration
            next_token_tensor = torch.LongTensor([[next_token_id]]).to(device)
            current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=-1)

            # Truncate if it exceeds the model's max_seq_len (safety check)
            if current_input_ids.shape[1] > model.model.max_seq_len:
                current_input_ids = current_input_ids[:, -model.model.max_seq_len:]
                # Also need to update generated_token_ids if we're doing hard truncation,
                # but for generation, we usually just let `generated_token_ids` grow
                # and `current_input_ids` will eventually be truncated at the next iteration if needed
                # For this simple greedy generation, let's just break if it gets too long
                # to avoid complexity with `generated_token_ids` truncation logic.
                if current_input_ids.shape[1] == model.model.max_seq_len:
                     break # Stop if we hit the max sequence length

        # Convert the generated token IDs back into human-readable text
        special_token_ids_to_filter = {pad_id, cls_id, sep_id, mask_id, unk_id}
        output_words = [
            extended_idx_to_token.get(token_id, '[UNK]') # Use .get with a default for safety
            for token_id in generated_token_ids
            if token_id not in special_token_ids_to_filter
        ]
        return " ".join(output_words)

