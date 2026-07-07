import torch
from nuscenes.nuscenes import NuScenes
from model import BEVFusionModel
from torch.utils.data import  DataLoader
from dataset import NuScenesDataset
from collator import collate_fn
import torch.nn as nn
import torch.optim as optim
from train import train_one_epoch
from evaluate import evaluate_model
from inference import run_inference

IMG_FEAT_DIM = 512 # Output from CameraBackbone
BEV_H, BEV_W = 200, 200 # Final BEV map dimensions
CAMERA_BEV_CHANNELS = 256
LIDAR_BEV_CHANNELS = 256
FUSED_BEV_CHANNELS = 256
DEPTH_CHANNELS = 64
NUM_CAMERAS = 6
LIDAR_GRID_SIZE = (BEV_W, BEV_H, 10) # (X_DIM, Y_DIM, Z_DIM)

BATCH_SIZE = 16 # For demonstration, use a small batch size
NUM_WORKERS = 0 # Set to 0 for debugging, increase for faster data loading
NUM_EPOCHS = 1 # Number of training epochs
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Initialize NuScenes object ---
# Define the dataroot path where the nuScenes dataset is located
dataroot = "C:/Users/SANGA/Downloads/v1.0-mini"

nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)

# --- 2. Instantiate Datasets and DataLoaders ---
print("\nSetting up Datasets and DataLoaders...")
# Create training dataset (using 'mini_train' split)
train_dataset = NuScenesDataset(
    nusc=nusc, 
    version='v1.0-mini', 
    dataroot=dataroot,
    image_size=(224, 224), # Example, match your CameraBackbone input
    bev_range=([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]),
    voxel_size=(0.1, 0.1, 0.2),
    max_points_per_voxel=32,
    num_cameras=NUM_CAMERAS,
    split='mini_train'
)

val_dataset = NuScenesDataset(
    nusc=nusc, 
    version='v1.0-mini', 
    dataroot=dataroot,
    image_size=(224, 224), # Example, match your CameraBackbone input
    bev_range=([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]),
    voxel_size=(0.1, 0.1, 0.2),
    max_points_per_voxel=32,
    num_cameras=NUM_CAMERAS,
    split='mini_val'
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn, 
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, # No need to shuffle validation data
    collate_fn=collate_fn, 
    num_workers=NUM_WORKERS
)

print(f"Training DataLoader has {len(train_loader)} batches.")
print(f"Validation DataLoader has {len(val_loader)} batches.")

# --- 3. Instantiate Model, Optimizer, and Loss Function ---
print("\nInstantiating BEVFusion Model...")
bevfusion_model = BEVFusionModel(
    bev_h=BEV_H, 
    bev_w=BEV_W,
    camera_feat_dim=IMG_FEAT_DIM,      # Map from your IMG_FEAT_DIM (512)
    lidar_feat_dim=LIDAR_BEV_CHANNELS, # Map to internal lidar tracking
    combined_feat_dim=FUSED_BEV_CHANNELS, # Maps to camera/lidar BEV target dim
    num_cameras=NUM_CAMERAS,
    depth_channels=DEPTH_CHANNELS,
    voxel_size=(0.5, 0.5, 0.2)         # Matches the calculated voxel size from your log
).to(device)

optimizer = optim.Adam(bevfusion_model.parameters(), lr=LEARNING_RATE)

criterion = nn.MSELoss() 

print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(NUM_EPOCHS,bevfusion_model, train_loader, optimizer, criterion, epoch, device)
    val_loss = evaluate_model(bevfusion_model, val_loader, criterion, device)
    
    # You might want to save the model weights here based on validation performance
    # torch.save(bevfusion_model.state_dict(), f'bevfusion_model_epoch_{epoch+1}.pth')

print("\n--- Training Complete ---")

if len(val_dataset) > 0:
    sample_idx = 0
    single_sample = val_dataset[sample_idx]
    print(f"Running inference on sample token: {single_sample['sample_token']}")

    inference_predictions, inference_fused_bev = run_inference(bevfusion_model, single_sample, device)
    
    print(f"Inference raw predictions shape: {inference_predictions.shape}")
    print(f"Inference fused BEV features shape: {inference_fused_bev.shape}")
else:
    print("Validation dataset is empty, skipping inference demonstration.")

print("\nTraining, Evaluation, and Inference pipeline demonstrated.")
print("Remember to fill in the TODOs in NuScenesDataset and the loss/metric calculations!")