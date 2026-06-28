import torch

def generatetext(model, tokenizer, prompt_text, max_new_tokens=50, segment_ids_list=None, eos_token_id=None, device=None):
    """
    Generates text using the CustomGenerativeTransformer model.

    Args:
        model: The CustomGenerativeTransformer model instance.
        tokenizer: A tokenizer with encode and decode methods.
        prompt_text (str): The initial text prompt.
        max_new_tokens (int): Maximum number of new tokens to generate.
        segment_ids_list (list, optional): List of segment IDs for the prompt.
                                         If None, a default (e.g., all 0s) is used.
        eos_token_id (int, optional): The end-of-sequence token ID.
                                    Defaults to tokenizer.eos_token_id if not provided.
        device (torch.device, optional): The device to run generation on.
                                       Defaults to model's device or CPU.

    Returns:
        str: The generated text.
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode the prompt text to token IDs
    initial_token_ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long, device=device)

    # Prepare segment IDs
    if segment_ids_list is None:
        # Default to segment ID 0 for all prompt tokens
        segment_ids = torch.zeros_like(initial_token_ids, dtype=torch.long, device=device)
    else:
        if len(segment_ids_list) != len(initial_token_ids):
            raise ValueError("segment_ids_list must have the same length as initial_token_ids.")
        segment_ids = torch.tensor(segment_ids_list, dtype=torch.long, device=device)

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    print(f"Initial prompt tokens (IDs): {initial_token_ids.tolist()}")
    print(f"Initial segment IDs: {segment_ids.tolist()}")
    print(f"Generating up to {max_new_tokens} new tokens...")

    # Use the model's generate method
    generated_ids = model.generate(
        initial_token_ids=initial_token_ids,
        segment_ids=segment_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id
    )

    # Decode the generated token IDs back to text
    generated_text = tokenizer.decode(generated_ids.tolist())

    return generated_text