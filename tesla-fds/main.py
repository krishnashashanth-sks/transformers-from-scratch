import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os # For checkpointing

# --- 1. Instantiate FSDSystem model and FSDLoss calculator, move to device ---

# Define comprehensive parameters for all modules, consistent with previous FSDSystem example
# (Copied from the last successful FSDSystem example usage)
params = {
    # Common
    'batch_size': 2,
    'num_cameras': 4,
    'bev_h': 10, # Reduced for faster demonstration
    'bev_w': 10, # Reduced for faster demonstration
    'num_frames_to_fuse': 3, # T: current + 2 history

    # Camera Encoder
    'cam_img_in_channels': 3, # RGB
    'cam_img_height': 224,
    'cam_img_width': 224,
    'cam_backbone_out_channels': 256,
    'cam_fpn_out_channels': 128,
    'cam_bev_channels': 64, # Channels per camera after ViewTransformer

    # LiDAR Encoder
    'lidar_voxel_in_channels': 4, # x,y,z,intensity
    'lidar_voxel_z_dim': 8,
    'lidar_voxel_xy_dim': 20, # XY dimensions of voxel grid
    'lidar_bev_channels': 64,

    # Radar Encoder
    'radar_in_channels': 2, # e.g., range, doppler
    'radar_raw_h': 128,
    'radar_raw_w': 128,
    'radar_bev_channels': 64,

    # MultiModalTemporalFusion
    'fusion_embed_dim': 128,
    'fusion_num_heads': 8,
    'fusion_num_layers': 2,
    'fusion_dropout': 0.1,

    # Occupancy Network
    'occupancy_query_point_dim': 3,
    'num_query_points': 1000,
    'occupancy_hidden_dim': 128,
    'occupancy_output_dim': 1,

    # Object Detection Head
    'det_num_classes': 10,
    'det_num_regression_params': 9,
    'det_hidden_dim': 128,

    # Semantic Segmentation Head
    'seg_num_semantic_classes': 5,
    'seg_hidden_dim': 128,

    # Depth Estimation Head
    'depth_hidden_dim': 128,

    # Agent-Centric Feature Extractor
    'agent_input_features_dim': 10, # e.g., (x_norm, y_norm, z, dx, dy, dz, yaw, vx, vy, class_id_one_hot)
    'agent_bev_patch_size': 5,
    'agent_feature_extractor_output_dim': 64,
    'max_num_agents': 5, # N_agents: max agents in a frame

    # Temporal Encoder
    'temporal_encoder_max_frames': 10,
    'temporal_encoder_embed_dim': 128,
    'temporal_encoder_num_heads': 8,
    'temporal_encoder_num_layers': 2,
    'temporal_encoder_dropout': 0.1,

    # Interaction Model
    'interaction_model_embed_dim': 128,
    'interaction_model_max_agents': 20,
    'interaction_model_num_heads': 8,
    'interaction_model_num_layers': 2,
    'interaction_model_dropout': 0.1,

    # Trajectory Decoder
    'num_future_steps_pred': 30, # Future trajectory length (prediction)
    'num_modes': 6, # Number of predicted trajectories per agent
    'trajectory_point_dim_pred': 2, # (x, y) for each point in predicted trajectory
    'trajectory_decoder_hidden_dim': 256,

    # Planning Module
    'ego_state_dim': 10, # Example: (x, y, yaw, vx, vy, ax, ay, speed, acceleration, jerk)
    'planning_context_embedding_dim': 256,
    'num_high_level_actions': 5, # e.g., keep lane, turn left, turn right, stop, change lane
    'num_candidate_trajectories': 10,
    'num_future_steps_plan': 50, # Longer planning horizon
    'trajectory_point_dim_plan': 2, # (x, y) for each point in planned trajectory
    'planning_hidden_dim': 256,
}


# Instantiate FSDSystem model
model = FSDSystem(**params)

# Instantiate the FSDLoss calculator
loss_calculator = FSDLoss()

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_calculator.to(device)

print(f"Model and Loss Calculator moved to: {device}")

# Optimizer (AdamW)
learning_rate = 1e-4
weight_decay = 1e-2 # Standard value for weight decay with AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print(f"Optimizer: {optimizer.__class__.__name__} initialized with LR={learning_rate}, Weight Decay={weight_decay}")

# Learning Rate Scheduler (Cosine Annealing with Warm-up)
num_epochs = 5 # Reduced for quicker demo of evaluation
warmup_epochs = 1
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    return 1.0
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, cosine_scheduler]) # Correct usage for ChainedScheduler
print(f"Learning Rate Scheduler: ChainedScheduler (Warmup for {warmup_epochs} epochs, then Cosine Annealing)")

# Mixed-precision scaler
scaler = torch.cuda.amp.GradScaler()
print("torch.cuda.amp.GradScaler initialized for mixed-precision training.")

max_grad_norm = 1.0 # Gradient clipping maximum norm

# Create dummy DataLoaders for training and validation
num_train_samples = 20 # More samples for a slightly longer demo
num_val_samples = 10
train_dataset = DummyFSDDataset(num_train_samples, params)
val_dataset = DummyFSDDataset(num_val_samples, params)

# Adjust batch_size for DataLoader, params['batch_size'] from setup was 2.
train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

print(f"Training with {len(train_dataloader)} batches, Validating with {len(val_dataloader)} batches per epoch.")

# Run the training process
trained_model, final_best_metric = train_fsd_system(
    model, train_dataloader, val_dataloader, loss_calculator,
    optimizer, scheduler, scaler, device, num_epochs, max_grad_norm,
    params # Pass params to training function for evaluation
)

print(f"\nTraining completed. Best validation metric achieved: {final_best_metric:.4f}")