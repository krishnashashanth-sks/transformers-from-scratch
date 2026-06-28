import torch
from utils import prepare_bert_input
from dataset import pad_id,cls_id,sep_id,mask_id,idx_to_token

def generate_text_from_mask(model, tokenize, prompt, max_len, device, num_predictions=1):
    model.eval()
    # Tokenize the prompt
    prompt_token_ids = tokenize(prompt) # Use the updated tokenize function

    # Prepare BERT input for the prompt
    input_ids, segment_ids, attention_mask = prepare_bert_input(
        prompt_token_ids, None, max_len, cls_id, sep_id, pad_id
    )

    # Convert to tensors
    input_ids_tensor = torch.LongTensor([input_ids]).to(device)
    segment_ids_tensor = torch.LongTensor([segment_ids]).to(device)
    attention_mask_tensor = torch.LongTensor([attention_mask]).unsqueeze(1).unsqueeze(2).to(device)

    # Find the positions of MASK tokens in the input
    mask_positions = [i for i, id in enumerate(input_ids) if id == mask_id]

    if not mask_positions:
        print("No [MASK] tokens found in the prompt (after tokenization). Cannot generate.")
        return ""

    generated_text = list(input_ids) # Start with the input IDs, including the [MASK] IDs

    with torch.no_grad():
        # Perform forward pass to get MLM predictions
        # Note: model returns mlm_prediction_scores, nsp_prediction_scores
        mlm_prediction_scores, _ = model(
            input_ids_tensor,
            segment_ids_tensor,
            attention_mask_tensor
        )

        # Get predictions for the masked positions
        for mp in mask_positions:
            # Get the logits for the masked position
            masked_logits = mlm_prediction_scores[0, mp, :]

            # Get the top-k predictions (or just the top-1 for simplicity)
            # Here we just take the single most probable token
            predicted_token_id = torch.argmax(masked_logits).item()
            generated_text[mp] = predicted_token_id

    # Convert the generated token IDs back to words
    output_words = [idx_to_token[id] for id in generated_text if id not in [cls_id, sep_id, pad_id]]
    return " ".join(output_words)