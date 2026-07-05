import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from collator import collate_fn
from dataset import SyntheticWorldDataset
from model import Full3DWorldGenerator
from train import train_model
from utils import *
import torch.nn as nn
import torch.optim as optim

# Define model parameters (consistent with previous module tests)
vocab_size = 10000
text_embed_dim = 256
text_num_heads = 4
text_num_layers = 2
text_dim_feedforward = 512
max_seq_len = 50

img_size = 256
patch_size = 16
in_channels = 4 # RGBA images

latent_dim_neRF = 128
multimodal_embed_dim = 256 # Should match text_embed_dim
time_emb_dim = 128
unet_num_blocks = 3
unet_channels_start = 64

nerf_hidden_dim = 128
nerf_output_density_dim = 1
nerf_output_color_dim = 3

memory_hidden_dim = 256
memory_num_layers = 1

L_pos = 10
L_dir = 4

# Instantiate the full model
full_model = Full3DWorldGenerator(
    vocab_size,
    text_embed_dim, text_num_heads, text_num_layers, text_dim_feedforward, max_seq_len,
    img_size, patch_size, in_channels,
    latent_dim_neRF, multimodal_embed_dim, time_emb_dim, unet_num_blocks, unet_channels_start,
    nerf_hidden_dim, nerf_output_density_dim, nerf_output_color_dim,
    memory_hidden_dim, memory_num_layers,
    L_pos=L_pos,
    L_dir=L_dir
)
latent_dim_neRF = 128 # The dimension of the NeRF latent vector
max_seq_len = 50    # Maximum sequence length for text tokenization
img_size = 256      # Resolution of images
in_channels = 4     # Number of channels for images (RGBA)

# Create a dataset instance
dataset = SyntheticWorldDataset(base_output_dir='synthetic_dataset', max_vocab_size=1000, min_freq=1)

# Create a DataLoader instance
batch_size = 2 # Assuming this is defined as a global variable elsewhere, if not, define it.
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

noise_prediction_loss_fn = nn.MSELoss()

rendering_loss_fn = nn.MSELoss()

optimizer = optim.AdamW(full_model.parameters(), lr=1e-4) # Learning rate is an example, will be tuned later

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
full_model.to(device)

dataset_for_training = SyntheticWorldDataset(base_output_dir='synthetic_dataset', max_vocab_size=1000, min_freq=1)
dataloader_for_training = DataLoader(
    dataset_for_training,
    batch_size=batch_size, # Use the batch_size defined in kernel state (e.g., 2)
    shuffle=True,
    collate_fn=collate_fn
)

# Training parameters
num_epochs = 10
log_interval = 2 # Log progress every N batches
# Training Loop
train_model(num_epochs,full_model,dataloader_for_training,optimizer,noise_prediction_loss_fn,batch_size,log_interval,device)


checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(full_model.state_dict(), os.path.join(checkpoint_dir, 'full_model_final.pth'))
print(f"Final model checkpoint saved to {os.path.join(checkpoint_dir, 'full_model_final.pth')}")
create_full_synthetic_dataset(num_scenes=5, base_output_dir='synthetic_eval_dataset')

# Instantiate SyntheticWorldDataset for evaluation
eval_dataset = SyntheticWorldDataset(base_output_dir='synthetic_eval_dataset', max_vocab_size=1000, min_freq=1)

# Instantiate DataLoader for the evaluation dataset
# Using a batch_size of 1 for evaluation to handle potential memory constraints and evaluate scenes individually.
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=False, # No need to shuffle for evaluation
    collate_fn=collate_fn
)

print("Evaluation SyntheticWorldDataset and DataLoader instantiated successfully.")

# Parameters for rendering
N_samples_render = 64 # Number of samples along each ray for rendering
H_render, W_render = img_size, img_size # Image dimensions for rendering
all_psnr_scores=[]
all_ssim_scores=[]

with torch.no_grad(): # Disable gradient calculations for evaluation
    for batch_idx, batch in enumerate(eval_dataloader):
        print(f"\n--- Evaluating Batch {batch_idx + 1}/{len(eval_dataloader)} ---")
        # 7. For each batch:
        # 7.1. Move inputs to the appropriate device
        tokenized_text = batch['tokenized_text'].to(device)
        ref_image = batch['ref_image'].to(device)

        # 7.2. Generate a NeRF latent representation (`generated_nerf_latent`) for the current scene
        # The full_model's forward method in eval mode returns a generated NeRF latent.
        # We need a dummy `timesteps` even in inference mode as it's part of the forward signature,
        # but it won't be used by the diffusion model's `sample` equivalent here.
        dummy_timesteps = torch.zeros(tokenized_text.size(0), dtype=torch.long, device=device) # Dummy timestep for inference
        generated_neRF_latent = full_model(
            text_input=tokenized_text,
            image_input=ref_image,
            timesteps=dummy_timesteps
        ) # Shape: (batch_size, latent_dim_neRF)

        # Ensure the generated_neRF_latent is treated as a single latent for rendering multiple views
        # For this setup, batch_size for eval_dataloader is 1, so generated_neRF_latent[0] gives the single latent.
        # If batch_size > 1, we would iterate through each latent in the batch.
        current_scene_generated_latent = generated_neRF_latent[0].unsqueeze(0) # (1, latent_dim_neRF)

        # 7.3. Iterate through the multi-view ground truth images
        nerf_images_batch = batch['nerf_images'] # (batch_size, max_nerf_views, C, H, W)
        nerf_extrinsics_batch = batch['nerf_extrinsics'] # (batch_size, max_nerf_views, 4, 4)
        nerf_intrinsic_batch = batch['nerf_intrinsic'] # dict of (batch_size,) tensors

        # Assuming batch_size is 1 for eval_dataloader
        gt_nerf_images = nerf_images_batch[0] # (max_nerf_views, C, H, W)
        nerf_extrinsics = nerf_extrinsics_batch[0] # (max_nerf_views, 4, 4)
        nerf_intrinsics = {k: v[0] for k, v in nerf_intrinsic_batch.items()} # dict of scalar tensors

        # We only consider actual views, not padded ones (padded with zeros).
        # Filter out padded images (where all channels are zero)
        # A simple heuristic: check if the sum of pixels is non-zero
        valid_views_mask = gt_nerf_images.sum(dim=[1,2,3]) > 0
        gt_nerf_images = gt_nerf_images[valid_views_mask]
        nerf_extrinsics = nerf_extrinsics[valid_views_mask]

        for view_idx in range(gt_nerf_images.shape[0]):
            gt_image_tensor = gt_nerf_images[view_idx].to(device) # (C, H, W)
            nerf_extrinsic_tensor = nerf_extrinsics[view_idx].to(device) # (4, 4)

            # 7.3.1. Generate rays
            ray_origins, ray_directions = get_rays(H_render, W_render, nerf_intrinsics, nerf_extrinsic_tensor)

            # 7.3.2. Render an image
            # current_scene_generated_latent needs to be repeated for the number of rays
            expanded_latent = current_scene_generated_latent.repeat(ray_origins.shape[0], 1)
            rendered_image_flat, _ = render_rays(
                full_model.nerf_mlp,
                expanded_latent,
                ray_origins,
                ray_directions,
                N_samples_render,
                L_pos, L_dir
            )
            rendered_image_tensor = rendered_image_flat.view(H_render, W_render, 3).permute(2, 0, 1) # (C, H, W)

            # 7.3.3. Convert to NumPy arrays and calculate metrics
            gt_image_np = gt_image_tensor[:3, :, :].permute(1, 2, 0).cpu().numpy() # (H, W, 3), ignore alpha for metrics
            rendered_image_np = rendered_image_tensor.permute(1, 2, 0).cpu().numpy() # (H, W, 3)

            # Ensure they are normalized to [0, 1] for PSNR/SSIM functions
            gt_image_np = np.clip(gt_image_np, 0, 1)
            rendered_image_np = np.clip(rendered_image_np, 0, 1)

            psnr = calculate_psnr(gt_image_np, rendered_image_np)
            ssim = calculate_ssim(gt_image_np, rendered_image_np)

            all_psnr_scores.append(psnr)
            all_ssim_scores.append(ssim)
            # print(f"  View {view_idx+1}: PSNR = {psnr:.2f}, SSIM = {ssim:.4f}")

# 8. Calculate and print average metrics
mean_psnr = np.mean(all_psnr_scores)
mean_ssim = np.mean(all_ssim_scores)

print(f"\nAverage PSNR over evaluation dataset: {mean_psnr:.2f}")
print(f"Average SSIM over evaluation dataset: {mean_ssim:.4f}")

# Parameters for rendering
N_samples_render = 64 # Number of samples along each ray for rendering
H_render, W_render = img_size, img_size # Image dimensions for rendering

# Select a few scenes from the evaluation dataset for visualization
num_examples_to_visualize = 3
selected_indices = random.sample(range(len(eval_dataset)), min(num_examples_to_visualize, len(eval_dataset)))

full_model.eval() # Set model to evaluation mode
full_model.to(device)

print(f"Visualizing {len(selected_indices)} examples from the evaluation dataset...")

with torch.no_grad(): # Disable gradient calculations for evaluation
    for i, idx in enumerate(selected_indices):
        print(f"\n--- Visualizing Example {i + 1}/{len(selected_indices)} (Scene Index: {idx}) ---")
        batch_item = eval_dataset[idx] # Get a single item from the dataset

        # Prepare batch-like inputs for the model (unsqueeze to add batch dimension)
        tokenized_text = batch_item['tokenized_text'].unsqueeze(0).to(device)
        ref_image = batch_item['ref_image'].unsqueeze(0).to(device)

        # Generate a NeRF latent representation (`generated_nerf_latent`)
        dummy_timesteps = torch.zeros(tokenized_text.size(0), dtype=torch.long, device=device) # Dummy timestep for inference
        generated_neRF_latent = full_model(
            text_input=tokenized_text,
            image_input=ref_image,
            timesteps=dummy_timesteps
        ) # Shape: (1, latent_dim_neRF)

        # Select one ground truth view for visualization (e.g., the first valid one)
        gt_nerf_images = batch_item['nerf_images'] # (num_views, C, H, W)
        nerf_extrinsics = batch_item['nerf_extrinsics'] # (num_views, 4, 4)
        nerf_intrinsics_dict = batch_item['nerf_intrinsic'] # dict of scalar tensors

        # Filter out padded images (if any) and take the first valid one
        valid_views_mask = gt_nerf_images.sum(dim=[1,2,3]) > 0
        valid_gt_nerf_images = gt_nerf_images[valid_views_mask]
        valid_nerf_extrinsics = nerf_extrinsics[valid_views_mask]

        if valid_gt_nerf_images.shape[0] == 0:
            print(f"  No valid ground truth images found for scene index {idx}. Skipping visualization.")
            continue

        # Take the first valid view for comparison
        gt_image_tensor = valid_gt_nerf_images[0].to(device) # (C, H, W)
        nerf_extrinsic_tensor = valid_nerf_extrinsics[0].to(device) # (4, 4)

        # Use camera intrinsics from the batch_item, converting dict values to device
        nerf_intrinsics = {k: v.to(device) for k, v in nerf_intrinsics_dict.items()}

        # Generate rays for the selected view
        ray_origins, ray_directions = get_rays(H_render, W_render, nerf_intrinsics, nerf_extrinsic_tensor)

        # Render an image from the generated NeRF latent
        # Repeat the single generated_neRF_latent for all rays
        expanded_latent = generated_neRF_latent.repeat(ray_origins.shape[0], 1)
        rendered_image_flat, _ = render_rays(
            full_model.nerf_mlp,
            expanded_latent,
            ray_origins,
            ray_directions,
            N_samples_render,
            L_pos, L_dir
        )
        rendered_image_tensor = rendered_image_flat.view(H_render, W_render, 3).permute(2, 0, 1) # (C, H, W)

        # Convert to NumPy arrays and normalize for PSNR/SSIM and plotting
        # Ground truth images usually have 4 channels (RGBA), but metrics are usually on RGB (first 3 channels)
        gt_image_np = gt_image_tensor[:3, :, :].permute(1, 2, 0).cpu().numpy() # (H, W, 3)
        rendered_image_np = rendered_image_tensor.permute(1, 2, 0).cpu().numpy() # (H, W, 3)

        gt_image_np = np.clip(gt_image_np, 0, 1)
        rendered_image_np = np.clip(rendered_image_np, 0, 1)

        psnr = calculate_psnr(gt_image_np, rendered_image_np)
        ssim = calculate_ssim(gt_image_np, rendered_image_np)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Scene {idx} | PSNR: {psnr:.2f}, SSIM: {ssim:.4f}", fontsize=14)

        axes[0].imshow(gt_image_np)
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')

        axes[1].imshow(rendered_image_np)
        axes[1].set_title("Model Rendered")
        axes[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()
