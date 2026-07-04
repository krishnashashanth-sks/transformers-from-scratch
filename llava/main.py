import torch
from model import LLaVA
from inference import llava_inference
import torch.nn as nn
import torch.optim as optim

llava_params = {
    'image_size': 224,
    'patch_size': 16,
    'in_channels': 3,
    'embed_dim': 768,
    'num_heads': 12,
    'num_layers': 6,
    'mlp_dim': 3072,
    'vocab_size': 10000,
    'max_seq_len': 512,
    'input_dim': 768, # Output of VisionEncoder and input for ProjectionLayer
    'output_dim': 768 # Output of ProjectionLayer and embed_dim for LanguageModel
}

model = LLaVA(
    vision_encoder_params=llava_params,
    projection_layer_params={}, # As per instructions, pass empty dictionaries
    language_model_params={}    # As per instructions, pass empty dictionaries
)

print("LLaVA model instantiated successfully:")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

batch_size = 4

# Create dummy pixel_values (image input)
pixel_values = torch.randn(
    batch_size,
    llava_params['in_channels'],
    llava_params['image_size'],
    llava_params['image_size']
)

# Create dummy input_ids (text input)
input_ids = torch.randint(
    0,
    llava_params['vocab_size'],
    (batch_size, llava_params['max_seq_len'])
)

# Create dummy target_ids (ground truth for language model output)
target_ids = torch.randint(
    0,
    llava_params['vocab_size'],
    (batch_size, llava_params['max_seq_len'])
)

print(f"Shape of dummy pixel_values: {pixel_values.shape}")
print(f"Shape of dummy input_ids: {input_ids.shape}")
print(f"Shape of dummy target_ids: {target_ids.shape}")

model.train()
optimizer.zero_grad()

output_logits = model(pixel_values, input_ids)

# Reshape output_logits and target_ids for loss calculation
output_logits_reshaped = output_logits.view(-1, llava_params['vocab_size'])
target_ids_reshaped = target_ids.view(-1)

loss = criterion(output_logits_reshaped, target_ids_reshaped)

loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")

inference_batch_size = 1

# Create dummy pixel_values (image input) for inference
inference_pixel_values = torch.randn(
    inference_batch_size,
    llava_params['in_channels'],
    llava_params['image_size'],
    llava_params['image_size']
)

# Create dummy input_ids (text input) for inference
inference_input_ids = torch.randint(
    0,
    llava_params['vocab_size'],
    (inference_batch_size, llava_params['max_seq_len'])
)

predicted_ids = llava_inference(model, inference_pixel_values, inference_input_ids)

print(f"Shape of predicted token IDs after inference: {predicted_ids.shape}")
print(f"Predicted token IDs (first sample, first 10 tokens): {predicted_ids[0, :10].tolist()}")
