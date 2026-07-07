import torch

def electra_inference(model, tokenizer, text, max_seq_len, mask_token_id, vocab_size, device=None):
    model.eval() # Set model to evaluation mode

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input text
    encoded_input = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    original_input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    token_type_ids = encoded_input['token_type_ids'].to(device)

    print(f"Original Input Text: {text}")
    print("Original Input Tokens (first 10):", tokenizer.convert_ids_to_tokens(original_input_ids[0, :10].tolist()))
    print("Original Input IDs (first 10):", original_input_ids[0, :10].tolist())

    # --- 1. Generator Inference: Predict masked tokens ---
    # Create a masked version of the input for the generator
    masked_input_ids_gen = original_input_ids.clone()
    # Mask a few non-special tokens randomly or at specific positions
    # For this example, let's mask a few specific tokens (not [CLS] or [SEP])
    mask_indices_gen = []
    for i in range(1, min(original_input_ids.shape[1] - 1, 5)): # Mask up to 4 non-special tokens
        if original_input_ids[0, i] not in tokenizer.all_special_ids:
            mask_indices_gen.append(i)
    mask_indices_gen = torch.tensor(mask_indices_gen, dtype=torch.long, device=device)

    if len(mask_indices_gen) > 0:
        masked_input_ids_gen[0, mask_indices_gen] = mask_token_id
    else:
        print("Could not find suitable tokens to mask for generator inference.")
        generator_results = {"predicted_masked_tokens": []}

    print("\nMasked Input Tokens for Generator (first 10):", tokenizer.convert_ids_to_tokens(masked_input_ids_gen[0, :10].tolist()))

    with torch.no_grad():
        if len(mask_indices_gen) > 0:
            generator_output_logits = model.generator(
                input_ids=masked_input_ids_gen,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            # Get predictions only for masked positions
            predicted_masked_token_ids = torch.argmax(generator_output_logits[0, mask_indices_gen], dim=-1)
            predicted_masked_tokens_str = tokenizer.convert_ids_to_tokens(predicted_masked_token_ids.tolist())
            generator_results = {
                "masked_indices": mask_indices_gen.tolist(),
                "predicted_token_ids": predicted_masked_token_ids.tolist(),
                "predicted_tokens_str": predicted_masked_tokens_str
            }
        else:
             generator_results = {"predicted_masked_tokens": []}

    print("Generator Predictions:", generator_results)

    # --- 2. Discriminator Inference: Identify replaced tokens ---
    # Create a corrupted input for the discriminator
    corrupted_input_ids_disc = original_input_ids.clone()
    # Replace a few non-special tokens with random ones
    replace_indices_disc = []
    for i in range(1, min(original_input_ids.shape[1] - 1, 5)): # Replace up to 4 non-special tokens
        if original_input_ids[0, i] not in tokenizer.all_special_ids and i not in mask_indices_gen: # Avoid already masked tokens
            replace_indices_disc.append(i)
    replace_indices_disc = torch.tensor(replace_indices_disc, dtype=torch.long, device=device)

    if len(replace_indices_disc) > 0:
        random_replacements = torch.randint(low=10, high=vocab_size - 10, size=(len(replace_indices_disc),), dtype=torch.long, device=device)
        corrupted_input_ids_disc[0, replace_indices_disc] = random_replacements
    else:
        print("Could not find suitable tokens to replace for discriminator inference.")
        discriminator_results = {"replaced_indices": [], "predictions_str": []}

    print("\nCorrupted Input Tokens for Discriminator (first 10):", tokenizer.convert_ids_to_tokens(corrupted_input_ids_disc[0, :10].tolist()))

    with torch.no_grad():
        discriminator_output_logits = model.discriminator(
            input_ids=corrupted_input_ids_disc,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        probabilities = torch.sigmoid(discriminator_output_logits)
        predictions = (probabilities > 0.5).int()

    if len(replace_indices_disc) > 0:
        predictions_for_replaced_tokens = predictions[0, replace_indices_disc]
        discriminator_results = {
            "replaced_indices": replace_indices_disc.tolist(),
            "predictions": predictions_for_replaced_tokens.tolist(), # 1 = replaced, 0 = original
            "predictions_str": ["REPLACED" if p == 1 else "ORIGINAL" for p in predictions_for_replaced_tokens.tolist()]
        }
    else:
        discriminator_results = {"replaced_indices": [], "predictions_str": []}

    print("Discriminator Predictions:", discriminator_results)

    return generator_results, discriminator_results