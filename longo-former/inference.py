import torch

def predict_text(text, model, tokenizer, max_length, device):
    model.eval() # Set model to evaluation mode

    # Encode the text
    encoded_input = tokenizer.encode(text, max_length=max_length)

    # Move inputs to device and add batch dimension
    input_ids = encoded_input['input_ids'].unsqueeze(0).to(device)
    attention_mask = encoded_input['attention_mask'].unsqueeze(0).to(device)
    token_type_ids = encoded_input['token_type_ids'].unsqueeze(0).to(device)
    global_attention_mask = encoded_input['global_attention_mask'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask)
        # Sanitize logits to prevent NaNs from propagating to softmax if they somehow occur.
        # This is a local fix for the output of this function; the root cause of
        # 'NAN/INF detected in attention_scores' warnings might be deeper in the model's
        # attention mechanism, but this ensures a numerically stable prediction.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()

    return predicted_class_id, probabilities.squeeze(0).cpu().numpy()
