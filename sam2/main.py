import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import  DataLoader
from collator import custom_collate_fn
from model import SAM2
from dataset import DummyDataset
from losses import CombinedLoss
import matplotlib.pyplot as plt
from train import train_model
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration for reduced dimensions ---
IMG_SIZE = 256 # Reduced image size
BATCH_SIZE = 1 # Reduced batch size
PATCH_SIZE = 16
EMBED_DIM = 768
IMAGE_FEATURE_GRID_SIZE = IMG_SIZE // PATCH_SIZE # 256 / 16 = 16
NUM_EPOCHS=5

# Instantiate the dummy dataset
dummy_train_dataset = DummyDataset(num_samples=10, img_size=IMG_SIZE)
dummy_val_dataset = DummyDataset(num_samples=4, img_size=IMG_SIZE) # Small validation set

# Create DataLoaders with custom collate_fn
dummy_train_dataloader = DataLoader(dummy_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
dummy_val_dataloader = DataLoader(dummy_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# Instantiate the SAM2 model with reduced dimensions
model = SAM2(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=3,
    embed_dim=EMBED_DIM,
    image_encoder_depth=2, # Reduced for faster testing
    num_heads=12,
    mlp_ratio=4.,
    qkv_bias=True,
    drop_rate=0.,
    attn_drop_rate=0.,
    num_point_labels=2,
    prompt_encoder_scale=4.0,
    num_prompt_transformer_blocks=2,
    num_mask_tokens=4,
    num_decoder_layers=2,
    output_upscale_factor=4,
    image_feature_grid_size=IMAGE_FEATURE_GRID_SIZE
)

# Instantiate the CombinedLoss function
loss_fn = CombinedLoss(weight_focal=1.0, weight_dice=1.0)

# Instantiate the AdamW optimizer with model parameters
learning_rate = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Instantiate the learning rate scheduler
T_max = 10 # Reduced for faster testing
etamin_lr = 1e-7
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=etamin_lr)

# --- Verify a single forward pass ---
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
model.to(device)

history=train_model(NUM_EPOCHS,model,dummy_train_dataloader,dummy_val_dataloader,optimizer,scheduler,loss_fn,device)

# Create a list of epoch numbers
epochs = range(1, NUM_EPOCHS + 1)

# Create a figure with three subplots
plt.figure(figsize=(15, 5))

# Subplot 1: Training and Validation Loss
plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
plt.plot(epochs, history['train_loss'], label='Training Loss')
plt.plot(epochs, history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Subplot 2: Validation IoU Score
plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd plot
plt.plot(epochs, history['val_iou'], label='Validation IoU', color='orange')
plt.title('Validation IoU Score')
plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.legend()
plt.grid(True)

# Subplot 3: Validation Dice Score
plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd plot
plt.plot(epochs, history['val_dice'], label='Validation Dice', color='green')
plt.title('Validation Dice Score')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()
plt.grid(True)

# Adjust layout and display plots
plt.tight_layout()
plt.show()


# Ensure the model is in evaluation mode
model.eval()

# Get a sample from the validation dataloader
# Reset the dataloader to get a new iteration from the start if needed
dummy_val_dataloader_iter = iter(dummy_val_dataloader)
sample_batch = next(dummy_val_dataloader_iter)

# Move data to the same device as the model
image = sample_batch['image'].to(device)
points = sample_batch['points'].to(device)
point_labels = sample_batch['point_labels'].to(device)
boxes = sample_batch['boxes'].to(device)
ground_truth_mask = sample_batch['ground_truth_mask'].to(device)
original_image_size = tuple(sample_batch['original_image_size'][0].cpu().numpy())
points_mask = sample_batch['points_mask'].to(device)
boxes_mask = sample_batch['boxes_mask'].to(device)

print(f"Performing inference on an image of size: {original_image_size}")

with torch.no_grad():
    predicted_mask_logits = model(
        image=image,
        points=points,
        point_labels=point_labels,
        boxes=boxes,
        original_image_size=original_image_size,
        points_mask=points_mask,
        boxes_mask=boxes_mask
    )

# Process the predicted mask
# Assuming num_mask_tokens > 1, take the first one or the one with highest score if a matching strategy was implemented.
# For simplicity, we'll take the first mask output.
predicted_mask = torch.sigmoid(predicted_mask_logits[:, 0, :, :]).squeeze().cpu().numpy()
predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8) * 255 # Convert to binary mask (0 or 255)

# Convert original image and ground truth mask to numpy for plotting
original_image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy() # (C, H, W) -> (H, W, C)
# Normalize image to [0, 1] for display if it's not already
original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min() + 1e-8)

ground_truth_mask_np = ground_truth_mask.squeeze().cpu().numpy() * 255 # (1, H, W) -> (H, W)

# Plotting
plt.figure(figsize=(18, 6))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(original_image_np)
plt.title('Original Image')
plt.axis('off')

# Original Image with Prompts
plt.subplot(1, 4, 2)
plt.imshow(original_image_np)
if points.numel() > 0:
    # Filter points by mask to show only actual points
    actual_points = points.squeeze(0).cpu().numpy()[points_mask.squeeze(0).cpu().numpy()]
    if actual_points.shape[0] > 0:
        plt.scatter(actual_points[:, 0], actual_points[:, 1], c='red', marker='o', s=50, label='Points')
if boxes.numel() > 0:
    # Filter boxes by mask to show only actual boxes
    actual_boxes = boxes.squeeze(0).cpu().numpy()[boxes_mask.squeeze(0).cpu().numpy()]
    for box in actual_boxes:
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                         edgecolor='blue', facecolor='none', lw=2))
    plt.text(actual_boxes[0, 0], actual_boxes[0, 1], 'Box', color='blue', fontsize=10, verticalalignment='top')
plt.title('Image with Prompts')
plt.axis('off')

# Ground Truth Mask
plt.subplot(1, 4, 3)
plt.imshow(ground_truth_mask_np, cmap='gray')
plt.title('Ground Truth Mask')
plt.axis('off')

# Predicted Mask
plt.subplot(1, 4, 4)
plt.imshow(predicted_mask_binary, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.tight_layout()
plt.show()
