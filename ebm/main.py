import matplotlib.pyplot as plt
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import CNN_EnergyFunction
from train import train_ebm_pcd
from sampler import langevin_sampler
from visualize import imshow

image_channels = 3  # e.g., for RGB images
image_size = 32     # e.g., 32x32 image

# 1. Define the number of dummy images
num_dummy_images_cnn = 1000

# 2. Create a tensor of dummy image data
# image_channels and image_size are available from previous cells (e.g., 3 and 32 respectively)
dummy_image_data_for_training = torch.randn(num_dummy_images_cnn, image_channels, image_size, image_size)

# 3. Create a TensorDataset
dummy_image_dataset = TensorDataset(dummy_image_data_for_training)

# 4. Create a PyTorch DataLoader
batch_size_cnn = 64
dummy_image_dataloader = DataLoader(dummy_image_dataset, batch_size=batch_size_cnn, shuffle=True)

# 1. Define the device for training (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Instantiate the CNN_EnergyFunction
# image_channels and image_size are available from previous cells
cnn_energy_model = CNN_EnergyFunction(image_channels, image_size)

# 3. Move the cnn_energy_model to the chosen device
cnn_energy_model.to(device)

# 4. Call the train_ebm_pcd function
print("Starting CNN EBM training with PCD...")
train_ebm_pcd(cnn_energy_model, dummy_image_dataloader, epochs=5, lr=1e-4)
print("CNN EBM training with PCD finished.")
# 1. Initialize a tensor of random noise
num_generated_images = 64 # Batch size for generated samples
initial_generated_samples_cnn = torch.randn(num_generated_images, image_channels, image_size, image_size).to(device)

# 2. Call the langevin_sampler function
# Use parameters as suggested in the instructions.
print(f"Generating {num_generated_images} image samples using Langevin Dynamics...")
generated_image_samples = langevin_sampler(cnn_energy_model, initial_generated_samples_cnn, n_steps=200, step_size=0.1, noise_scale=0.005)

# 3. Store the generated samples
# Ensure they are detached from the computation graph and moved to CPU
generated_image_samples_cpu = generated_image_samples.detach().cpu()

print(f"Shape of generated image samples: {generated_image_samples_cpu.shape}")

# Select a few images to display
num_display = min(generated_image_samples_cpu.shape[0], 16) # Display up to 16 images

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Generated Image Samples from CNN EBM', fontsize=16)

for i in range(num_display):
    ax = fig.add_subplot(4, 4, i + 1) # Create a 4x4 grid
    imshow(generated_image_samples_cpu[i])

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()