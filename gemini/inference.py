import torch

# Inference Function
def run_multimodal_inference(model, text_tokens, images, audio_spectrograms, videos,
                             text_gen_head, multimodal_clf_head, tool_head, text_max_seq_len):
    """
    Performs multimodal inference using the trained model and its output heads.

    Args:
        model (nn.Module): The MultimodalTransformer model.
        text_tokens (torch.Tensor): Input text tokens.
        images (torch.Tensor): Input images.
        audio_spectrograms (torch.Tensor): Input audio spectrograms.
        videos (torch.Tensor): Input videos.
        text_gen_head (nn.Module): The text generation output head.
        multimodal_clf_head (nn.Module): The multimodal classification output head.
        tool_head (nn.Module): The tool use output head.
        text_max_seq_len (int): The maximum sequence length for text tokens, used for slicing.

    Returns:
        tuple: A tuple containing (text_logits, clf_logits, tool_logits, arg_gen_output).
    """

    model.eval()

    with torch.no_grad(): # Disable gradient calculations
        multimodal_output = model(text_tokens, images, audio_spectrograms, videos)

        text_output_for_gen = multimodal_output[:, :text_max_seq_len, :]
        text_logits = text_gen_head(text_output_for_gen)

        cls_output_for_clf = multimodal_output[:, 0, :]
        clf_logits = multimodal_clf_head(cls_output_for_clf)

        tool_logits, arg_gen_output = tool_head(cls_output_for_clf)

    return text_logits, clf_logits, tool_logits, arg_gen_output


#  Inference Function (`perform_inference`)
def perform_inference(model, text_tokens, images, audio_spectrograms, videos, 
                      text_gen_head, multimodal_clf_head, tool_head, text_max_seq_len, device):
    model.eval()
    with torch.no_grad():
        text_tokens = text_tokens.to(device)
        images = images.to(device)
        audio_spectrograms = audio_spectrograms.to(device)
        videos = videos.to(device)

        # Reuse the run_multimodal_inference function
        inference_text_logits, inference_clf_logits, inference_tool_logits, inference_arg_gen_output = \
            run_multimodal_inference(
                model, text_tokens, images, audio_spectrograms, videos,
                text_gen_head, multimodal_clf_head, tool_head, text_max_seq_len
            )
    return inference_text_logits, inference_clf_logits, inference_tool_logits, inference_arg_gen_output
