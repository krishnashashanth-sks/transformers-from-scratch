import torch

def llava_inference(model, pixel_values, input_ids):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        output_logits = model(pixel_values, input_ids)

    # Apply softmax to get probabilities over the vocabulary
    probabilities = torch.softmax(output_logits, dim=-1)

    # Get the predicted token IDs (the token with the highest probability at each position)
    predicted_token_ids = torch.argmax(probabilities, dim=-1)

    return predicted_token_ids