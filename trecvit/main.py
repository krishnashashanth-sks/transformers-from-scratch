from model import TRecViT
import torch.optim as optim
import torch
from dataset import dataloader
import torch.nn as nn
from train import train_model
from inference import predict_class_label

image_size_val = (128, 128) # Matches resize in transforms_pipeline
patch_size_val = (16, 16)
in_channels_val = 3
embed_dim_val = 768 # Common embedding dimension for transformers
sequence_length = 16 # Number of frames per video sequence
num_frames_val = sequence_length # From previous data loader
input_dim_val = embed_dim_val # 768
hidden_dim_val = embed_dim_val # Can be same as input_dim or different
num_gru_layers = 1 # Number of GRU layers
dropout_rate_gru = 0.1
embed_dim_val = embed_dim_val # 768 from previous steps
num_attention_heads = 12 # Common for transformers
mlp_hidden_dim = embed_dim_val * 4 # Common practice: 4x embed_dim
encoder_dropout_rate = 0.1
attention_dropout = 0.0
num_transformer_layers = 6 # Common number of layers for vision transformers
num_classes = 10 # Example: for a 10-class classification task

model = TRecViT(
    image_size=image_size_val,         # (128, 128)
    patch_size=patch_size_val,         # (16, 16)
    in_channels=in_channels_val,       # 3
    embed_dim=embed_dim_val,           # 768
    num_frames=num_frames_val,         # 16
    num_gru_layers=num_gru_layers,     # 1
    num_attention_heads=num_attention_heads, # 12
    mlp_dim=mlp_hidden_dim,            # 3072
    num_transformer_layers=num_transformer_layers, # 6
    num_classes=num_classes,
    dropout_rate=encoder_dropout_rate, # 0.1
    attention_dropout_rate=attention_dropout # 0.0
)

loss_function = nn.CrossEntropyLoss()
print(f"Loss Function: {loss_function}")

# 3. Define the Optimizer (AdamW)
learning_rate = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
print(f"Optimizer: {optimizer}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 

step_size = 10
gamma = 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

history=train_model(model, dataloader, loss_function, optimizer, scheduler, device, num_epochs)

idx_to_class = {i: f"Class_{i}" for i in range(num_classes)} # num_classes was 10 in model definition

# Reset dataloader iterator to get a fresh batch
example_sequences, example_labels = next(iter(dataloader))

# Take the first video from the batch
single_video_input = example_sequences[0]
original_label = example_labels[0].item()

print(f"\nPerforming inference on a single video input (original label: {idx_to_class[original_label]})...")

predicted_label, confidence = predict_class_label(model, single_video_input, idx_to_class, device)

print(f"Predicted Class: {predicted_label} (Confidence: {confidence:.4f})")
print("Inference demonstration complete.")