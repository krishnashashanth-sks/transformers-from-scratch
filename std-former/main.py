import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import DummyMovingMNIST
from model import STDiT
from utils import get_cosine_schedule,p_sample_loop
import torch
import torch.nn as nn
import torch.optim as optim
from train import train_model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters (ensure consistency with previous steps)
img_size = 64
patch_size = 8
in_channels = 1
embed_dim = 768
num_frames = 10
tubelet_size = 2
depth = 4 # Use a smaller depth for faster training demo
num_heads = 12
mlp_ratio = 4.0
drop_rate = 0.0
attn_drop_rate = 0.0
drop_path_rate = 0.0

# Training hyperparameters
epochs = 5 # Reduced for demonstration
learning_rate = 1e-4
batch_size = 4
num_diffusion_timesteps = 1000 # Total diffusion steps for cosine schedule

# 1. Dataset and DataLoader setup
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = DummyMovingMNIST(
    num_samples=100, # A reasonable number of dummy samples
    sequence_length=num_frames, 
    image_size=img_size, 
    transform=transform
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0 # Set to 0 for debugging, can increase for faster real training
)
print(f"DataLoader prepared with {len(train_dataset)} samples and batch size {batch_size}.")

# 2. Initialize STDiT model
model = STDiT(
    img_size=img_size, patch_size=patch_size, in_channels=in_channels,
    embed_dim=embed_dim, num_frames=num_frames, tubelet_size=tubelet_size,
    depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
    attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate
).to(device)
print(f"STDiT model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M parameters.")

# 3. Optimizer and Loss Function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss() # Standard for noise prediction in diffusion models

# 4. Get diffusion schedule parameters
schedule_params = get_cosine_schedule(num_diffusion_timesteps)

# 5. Training Loop
train_model(epochs,train_dataloader,num_diffusion_timesteps,model,optimizer,loss_fn,schedule_params,device)


# Set model to evaluation mode for inference
model.eval()

# Diffusion schedule parameters
num_diffusion_timesteps_inference = 100 # Can use fewer steps for faster sampling if DDIM is implemented
schedule_params = get_cosine_schedule(num_diffusion_timesteps_inference)

# Define the desired shape of the output data for a single sample
output_shape = (1, num_frames, in_channels, img_size, img_size)

print(f"Starting inference (sampling) with output shape: {output_shape} and {num_diffusion_timesteps_inference} diffusion steps.")

# Generate a sample using the p_sample_loop
with torch.no_grad():
    generated_sample = p_sample_loop(model, output_shape, schedule_params, device=device)

print(f"Generated sample shape: {generated_sample.shape}")
print(f"Generated sample value range: [{generated_sample.min():.4f}, {generated_sample.max():.4f}]")

# Denormalize for visualization: [-1, 1] -> [0, 1]
visualizable_sample = (generated_sample.cpu() * 0.5) + 0.5

# Visualize a few frames from the generated sample
num_frames_to_display = min(5, visualizable_sample.shape[1])

plt.figure(figsize=(num_frames_to_display * 3, 3))
for i in range(num_frames_to_display):
    plt.subplot(1, num_frames_to_display, i + 1)
    frame = visualizable_sample[0, i] # Select batch 0, frame i
    if frame.shape[0] == 1: # If grayscale, squeeze channel dimension
        frame = frame.squeeze(0)
    plt.imshow(frame.numpy(), cmap='gray')
    plt.title(f'Generated Frame {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()
