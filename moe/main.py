from model import CustomGenerativeTransformer
import torch
from tokenizer import DummyTokenizer
from inference import generatetext

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

vocab_size_model = 1000
d_model_model = 256
nhead_model = 8
num_encoder_layers_model = 2
dim_feedforward_model = 512
num_experts_model = 4
top_k_model = 2
num_segments_model = 4

moe_transformer_model = CustomGenerativeTransformer(
    vocab_size=vocab_size_model,
    d_model=d_model_model,
    nhead=nhead_model,
    num_encoder_layers=num_encoder_layers_model,
    dim_feedforward=dim_feedforward_model,
    num_experts=num_experts_model,
    top_k=top_k_model,
    num_segments=num_segments_model
)

# Instantiate the dummy tokenizer
dummy_tokenizer = DummyTokenizer(vocab_size=vocab_size_model)

# Training is missing because of dataset is not available

print("--- Demonstrating `generatetext` function ---")

# Example usage:
prompt = "write a short poem about nature"

# For simplicity, let's assign segment ID 0 for prompt, and 1 for generated content
# This assumes our dummy tokenizer maps words to IDs from 3 onwards.
# The actual segment_ids_list would be determined by your data preparation pipeline.
# Here, we'll just use a single segment ID for the entire prompt for now.
# In a real prompt engineering scenario, you might have different segments for instruction, example, input.

# Create a simple segment ID list for the prompt (all 0s for initial example)
prompt_token_ids = dummy_tokenizer.encode(prompt)
example_segment_ids = [0] * len(prompt_token_ids)

# Ensure moe_transformer_model is on the correct device for generation
# (It was moved to `device` in cell 7672cc89's test block)

generated_output_text = generatetext(
    model=moe_transformer_model,
    tokenizer=dummy_tokenizer,
    prompt_text=prompt,
    max_new_tokens=10, # Generate only a few tokens for quick demo
    segment_ids_list=example_segment_ids,
    eos_token_id=dummy_tokenizer.eos_token_id,
    device=device
)

print(f"\nPrompt: {prompt}")
print(f"Generated Text: {generated_output_text}")

print("\n--- `generatetext` demonstration complete --- ")
