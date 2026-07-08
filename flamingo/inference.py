import torch

def generate_caption(model, image, tokenizer, max_new_tokens, max_seq_len, num_visual_tokens, device, start_text="A picture of"):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        visual_features = model.vision_encoder(image)
        visual_tokens = model.perceiver_sampler(visual_features)

        initial_token_ids = tokenizer.tokenize(start_text)

        img_placeholders = [tokenizer.get_img_token_id()] * num_visual_tokens

        current_text_tokens = img_placeholders + initial_token_ids

        processed_text_token_ids = tokenizer.pad(current_text_tokens, max_seq_len)
        lm_input_tokens = torch.tensor(processed_text_token_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_token_ids = initial_token_ids.copy()

        for _ in range(max_new_tokens):
            logits = model.language_model(lm_input_tokens, visual_tokens)
            next_token_logits = logits[:, -1, :]

            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            if next_token_id == tokenizer.pad_token_id or next_token_id == tokenizer.img_token_id:
                break

            generated_token_ids.append(next_token_id)

            current_text_tokens.append(next_token_id)
            if len(current_text_tokens) > max_seq_len:
                current_text_tokens = current_text_tokens[1:]

            lm_input_tokens = torch.tensor(tokenizer.pad(current_text_tokens, max_seq_len), dtype=torch.long, device=device).unsqueeze(0)

        generated_words = [tokenizer.reverse_vocab.get(token_id, '[UNK]') for token_id in generated_token_ids]
        return " ".join(generated_words)
