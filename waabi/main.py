import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from collator import custom_collate_fn
from train import train_epoch
from evaluate import validate_epoch
from losses import EndToEndLoss,PerceptionLoss,PlanningLoss,PredictionLoss,FocalLoss,SmoothL1Loss,CrossEntropyLossSeg
from model import EndToEndModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Perception DataLoader
perception_dataset_train = CustomDataset(num_samples=100, task_type='perception')
perception_dataloader_train = DataLoader(
    perception_dataset_train,
    batch_size=1,
    shuffle=True,
    num_workers=0, # Use 0 for simplicity, set >0 in production
    collate_fn=custom_collate_fn
)

# Prediction DataLoader
prediction_dataset_train = CustomDataset(num_samples=100, task_type='prediction')
prediction_dataloader_train = DataLoader(
    prediction_dataset_train,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# Planning DataLoader
planning_dataset_train = CustomDataset(num_samples=100, task_type='planning')
planning_dataloader_train = DataLoader(
    planning_dataset_train,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# --- 3. Instantiate Components and Run End-to-End Training/Validation ---

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- DataLoaders ---
# Use CustomDataset with task_type='e2e' to get all required GTs
e2e_dataset_train = CustomDataset(num_samples=100, task_type='e2e')
e2e_dataloader_train = DataLoader(
    e2e_dataset_train,
    batch_size=2, # Using batch_size 2 for E2E
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn
)
# --- Models ---
# Define parameters for each sub-module. These should be consistent with their definitions.
perception_params = {
    'camera_in_channels': 3, 'camera_image_width': 640, 'camera_image_height': 480, 'camera_feature_dim': 512,
    'lidar_num_points': 1000, 'lidar_features_per_point': 4, 'lidar_feature_dim': 512,
    'radar_features_per_detection': 3, 'radar_feature_dim': 512,
    'output_bev_channels': 64, 'num_classes_boxes': 3, 'num_lane_points': 50
}

prediction_params = {
    'agent_state_dim': 7, 'historical_steps': 10, 'agent_hidden_dim': 128,
    'bev_channels': 5, 'bev_height': 10, 'bev_width': 10, # Corrected to 5 channels for one-hot encoding input
    'context_feature_dim': 256,
    'interaction_hidden_dim': 256, 'interaction_num_heads': 4,
    'num_output_trajectories': 3, 'prediction_horizon': 20,
    'pose_dim': 3, 'num_intention_classes': 5,
    'ego_feature_dim': 6
}

planning_params = {
    'bev_channels': 5, 'bev_height': 10, 'bev_width': 10, # Aligned with prediction's ContextEncoder input
    'num_agents': 3, 'num_output_trajectories': 3, 'prediction_horizon': 20, 'pose_dim': 3, 'num_intention_classes': 5,
    'ego_state_dim': 6,
    'target_trajectory_len': 30, 'control_dim': 3,
    'hidden_dim': 256
}


e2e_model = EndToEndModel(perception_params, prediction_params, planning_params).to(device)

# --- Loss Functions ---
# Define the weights for combining perception, prediction, and planning losses

perception_losses = {
    'bbox_cls_loss': FocalLoss(alpha=0.01, gamma=2),
    'bbox_reg_loss': SmoothL1Loss(beta=1.0),
    'semantic_loss': CrossEntropyLossSeg(), # Assuming multi-class segmentation
    'lane_loss': SmoothL1Loss(beta=1.0) # For lane coordinates
}

prediction_loss_weights = {
    'trajectory_loss': 1.0, # Weight for the combined trajectory regression + confidence loss
    'intention_loss': 0.5   # Weight for intention classification
}

# Define weighting hyperparameters for total perception loss
# These values would be tuned during experimentation
perception_loss_weights = {
    'bbox_cls_loss': 1.0,
    'bbox_reg_loss': 2.0, # Regression often needs higher weight
    'semantic_loss': 1.0,
    'lane_loss': 1.0
}
planning_loss_weights = {
    'optimal_trajectory_loss': 1.0, # Weight for trajectory prediction
    'control_command_loss': 1.0     # Weight for control commands
}

e2e_loss_weights = {
    'perception_loss': 1.0,
    'prediction_loss': 1.0,
    'planning_loss': 1.0
}

e2e_loss_fn = EndToEndLoss(
    perception_loss_fn=PerceptionLoss(perception_losses, perception_loss_weights),
    prediction_loss_fn=PredictionLoss(prediction_loss_weights, pose_dim=3, beta=1.0, num_agents_for_dataset=e2e_dataset_train.agent_templates.__len__()),
    planning_loss_fn=PlanningLoss(planning_loss_weights, pose_dim=3, control_dim=3, beta=1.0),
    e2e_loss_weights=e2e_loss_weights
)

# --- Optimizer and Scheduler ---
optimizer_e2e = torch.optim.AdamW(e2e_model.parameters(), lr=1e-3, weight_decay=1e-4)
print("Optimizer (AdamW) for EndToEndModel instantiated.")

num_epochs_e2e = 5 # Reduced epochs for conceptual run speed
scheduler_e2e = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_e2e, T_max=num_epochs_e2e)
print("Learning rate scheduler (CosineAnnealingLR) for EndToEndModel instantiated.")

# --- Start End-to-End Training and Validation Loop ---

for epoch in range(num_epochs_e2e):
    print(f"\nEpoch {epoch+1}/{num_epochs_e2e}")

    # Training Phase
    train_results_e2e = train_epoch(
        model=e2e_model,
        dataloader=e2e_dataloader_train,
        optimizer=optimizer_e2e,
        loss_function=e2e_loss_fn,
        device=device,
        task_type='e2e' # Specify 'e2e' task type
    )
    print(f"Train Total Loss: {train_results_e2e['total_loss']:.4f}")
    print(f"  Perception Loss: {train_results_e2e['perception_total_loss']:.4f}, Prediction Loss: {train_results_e2e['prediction_total_loss']:.4f}, Planning Loss: {train_results_e2e['planning_total_loss']:.4f}")

    # Validation Phase (using same dataloader for simplicity)
    val_results_e2e = validate_epoch(
        model=e2e_model,
        dataloader=e2e_dataloader_train, # Use a separate val_dataloader in production
        loss_function=e2e_loss_fn,
        device=device,
        task_type='e2e' # Specify 'e2e' task type
    )
    print(f"Validation Total Loss: {val_results_e2e['total_loss']:.4f}")
    print(f"  Perception Loss: {val_results_e2e['perception_total_loss']:.4f}, Prediction Loss: {val_results_e2e['prediction_total_loss']:.4f}, Planning Loss: {val_results_e2e['planning_total_loss']:.4f}")

    # Step the scheduler
    scheduler_e2e.step()
    print(f"Learning Rate: {optimizer_e2e.param_groups[0]['lr']:.6f}")

# 1. Set the e2e_model to evaluation mode and move to device
e2e_model.eval()
e2e_model.to(device)

# 2. Get a sample batch of data from the e2e_dataloader_train
# Iterator over the DataLoader once to get a sample batch
for sample_batch_idx, sample_data in enumerate(e2e_dataloader_train):
    if sample_batch_idx == 0:
        break # Just take the first batch

# 3. Move all relevant input tensors to the active device
# Inputs for EndToEndModel: camera_input, lidar_input, radar_input, ego_vehicle_state_features
camera_input_sample = sample_data['camera_input'].to(device)
lidar_input_sample = sample_data['lidar_input'].to(device)
radar_input_sample = sample_data['radar_input'].to(device)
ego_vehicle_state_features_sample = sample_data['ego_vehicle_state_features'].to(device)

# 4. Perform a forward pass through the e2e_model
with torch.no_grad(): # Disable gradient calculation for inference
    model_outputs = e2e_model(
        camera_input_sample,
        lidar_input_sample,
        radar_input_sample,
        ego_vehicle_state_features_sample
    )


# --- Visualization ---

# Extract ground truth for comparison (optional, but good for context)
gt_optimal_trajectory_sample = sample_data['gt_optimal_trajectory'][0].cpu().numpy() # [0] for first item in batch

# 5. Extract original camera image input
# Convert from CHW (tensor) to HWC (numpy) and unnormalize (0-1 to 0-255)
original_camera_image = (sample_data['camera_input'][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

# 6. Visualize the camera input
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(original_camera_image)
plt.title('Raw Camera Input')
plt.axis('off')

# 7. Extract and visualize the semantic BEV map
# Semantic map from perception_outputs: (Batch, Classes, H, W)
# Convert to (H, W) class index map for visualization
semantic_map_output = model_outputs['perception_outputs']['semantic_map'][0].cpu() # Get first item in batch
semantic_map_classes = torch.argmax(semantic_map_output, dim=0).numpy() # (H, W)

# Define a colormap for semantic map classes
# Class IDs: 0-background, 1-lane, 2-vehicle, 3-pedestrian, 4-traffic_light
colors = ['black', 'blue', 'red', 'green', 'yellow'] # Background, Lane, Vehicle, Pedestrian, Traffic Light
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(-0.5, len(colors)), cmap.N)

plt.subplot(1, 3, 2)
plt.imshow(semantic_map_classes, cmap=cmap, norm=norm)
plt.title('Semantic BEV Map (Perception Output)')
plt.colorbar(ticks=np.arange(len(colors)), format=plt.FuncFormatter(lambda i, *args: colors[int(i)]))
plt.axis('off')

# 8. Visualize predicted agent trajectories and ego-vehicle's optimal planned trajectory
# Use a scatter plot for BEV coordinates for simplicity
plt.subplot(1, 3, 3)

# Ego-vehicle's current position for context (from input ego_vehicle_state_features_sample)
ego_x = ego_vehicle_state_features_sample[0, 0].item()
ego_y = ego_vehicle_state_features_sample[0, 1].item()
plt.scatter(ego_x, ego_y, color='cyan', marker='P', s=200, label='Ego Vehicle (Current)')

# Predicted Agent Trajectories (Prediction Output)
# model_outputs['prediction_outputs']['predicted_trajectories']: (B, N_agents, K, T, pose_dim)
predicted_agent_trajectories = model_outputs['prediction_outputs']['predicted_trajectories'][0].cpu().numpy() # First item in batch
num_agents_predicted = predicted_agent_trajectories.shape[0]
num_modes = predicted_agent_trajectories.shape[1]

for i in range(num_agents_predicted):
    for k in range(num_modes):
        # Plot x, y for each trajectory point
        plt.plot(predicted_agent_trajectories[i, k, :, 0], predicted_agent_trajectories[i, k, :, 1],
                 color='orange', linestyle='--', alpha=0.5, label=f'Agent {i} Pred Traj {k}' if i==0 and k==0 else "")

# Ego-vehicle Optimal Planned Trajectory (Planning Output)
# model_outputs['planning_outputs']['optimal_trajectory']: (B, target_trajectory_len, pose_dim)
ego_planned_trajectory = model_outputs['planning_outputs']['optimal_trajectory'][0].cpu().numpy()
plt.plot(ego_planned_trajectory[:, 0], ego_planned_trajectory[:, 1], color='green', linewidth=2, label='Ego Planned Trajectory')

# Ground Truth Optimal Trajectory (for comparison if available)
if gt_optimal_trajectory_sample is not None:
    plt.plot(gt_optimal_trajectory_sample[:, 0], gt_optimal_trajectory_sample[:, 1], color='purple', linestyle=':', linewidth=2, label='Ego GT Trajectory')

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Predicted & Planned Trajectories (BEV)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
