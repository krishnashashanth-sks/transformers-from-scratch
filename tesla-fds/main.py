from model import FSDSystem
from train import train_fsd_system
from losses import FSDLoss
import torch
from torch.utils.data import DataLoader
from dataset import DummyFSDDataset
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Instantiate FSDSystem model and FSDLoss calculator, move to device ---

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


# Put model in evaluation mode
model.eval()

print("\n--- Generating dummy inputs for visualization --- ")
# Generate a single set of dummy inputs (batch_size=1 for easier visualization)
vis_batch_size = 1

dummy_cam_input_sequence = torch.randn(
    vis_batch_size, params['num_frames_to_fuse'], params['num_cameras'],
    params['cam_img_in_channels'], params['cam_img_height'], params['cam_img_width']
).to(device)
dummy_lidar_input_sequence = torch.randn(
    vis_batch_size, params['num_frames_to_fuse'], params['lidar_voxel_in_channels'],
    params['lidar_voxel_z_dim'], params['lidar_voxel_xy_dim'], params['lidar_voxel_xy_dim']
).to(device)
dummy_radar_input_sequence = torch.randn(
    vis_batch_size, params['num_frames_to_fuse'], params['radar_in_channels'],
    params['radar_raw_h'], params['radar_raw_w']
).to(device)
dummy_occupancy_query_points = (torch.rand(vis_batch_size, params['num_query_points'], params['occupancy_query_point_dim']) * 2 - 1).to(device)

dummy_detected_agents_states_seq = (torch.rand(
    vis_batch_size, params['num_frames_to_fuse'], params['max_num_agents'], params['agent_input_features_dim']
) * 2 - 1).to(device)
dummy_detected_agents_states_seq[:, :, :, 0] = dummy_detected_agents_states_seq[:, :, :, 0] * 2 - 1
dummy_detected_agents_states_seq[:, :, :, 1] = dummy_detected_agents_states_seq[:, :, :, 1] * 2 - 1

dummy_ego_vehicle_state = torch.randn(vis_batch_size, params['ego_state_dim']).to(device)


print("--- Performing forward pass through FSDSystem for visualization --- ")
with torch.no_grad():
    full_system_outputs = model(
        cam_input_sequence=dummy_cam_input_sequence,
        lidar_input_sequence=dummy_lidar_input_sequence,
        radar_input_sequence=dummy_radar_input_sequence,
        occupancy_query_points=dummy_occupancy_query_points,
        detected_agents_states_seq=dummy_detected_agents_states_seq,
        ego_vehicle_state=dummy_ego_vehicle_state
    )

print("--- Visualizing FSDSystem Outputs --- ")

# Extract outputs for the first item in the batch
perception_outputs = {k: v[0].cpu().numpy() if isinstance(v, torch.Tensor) else {vk: vv[0].cpu().numpy() for vk, vv in v.items()} for k, v in full_system_outputs['perception_outputs'].items()}
prediction_outputs = {k: v[0].cpu().numpy() for k, v in full_system_outputs['prediction_outputs'].items()}
planning_outputs = {k: v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in full_system_outputs['planning_outputs'].items()}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('FSD System Outputs Visualization (Single Frame)', fontsize=16)

# --- Perception Visualization ---

# 1. Semantic Segmentation
semantic_seg_map = np.argmax(perception_outputs['semantic_segmentation'], axis=0) # Convert logits to class index
axes[0, 0].imshow(semantic_seg_map, cmap='Paired', origin='lower')
axes[0, 0].set_title('Perception: Semantic Segmentation (BEV)')
axes[0, 0].set_ylabel('Y-axis (BEV)')
axes[0, 0].set_xlabel('X-axis (BEV)')

# 2. Depth Map
depth_map = perception_outputs['depth_map'].squeeze()
axes[0, 1].imshow(depth_map, cmap='viridis', origin='lower')
axes[0, 1].set_title('Perception: Depth Map (BEV)')
axes[0, 1].set_ylabel('Y-axis (BEV)')
axes[0, 1].set_xlabel('X-axis (BEV)')

# 3. Object Detections (simplified to showing centers on BEV)
# Assuming object_detections['reg_outputs'] has cx, cy at indices 0, 1
# And obj_mask indicates valid detections
obj_detection_reg = perception_outputs['object_detections']['reg_outputs']
obj_detection_cls_logits = perception_outputs['object_detections']['logits']

# To get object locations, we can threshold classification logits and get center from regression outputs
# For simplicity, let's just show some prominent detections based on class score
# We'll plot top N detections based on class score (highest logit for any class)
scores = np.max(obj_detection_cls_logits, axis=0)
y_coords, x_coords = np.where(scores > np.percentile(scores, 95)) # Top 5% scoring locations

axes[0, 2].imshow(semantic_seg_map, cmap='Paired', origin='lower') # Use semantic map as background
if len(x_coords) > 0:
    axes[0, 2].scatter(x_coords, y_coords, color='red', marker='o', s=50, label='Detected Objects')
axes[0, 2].set_title('Perception: Object Detections (BEV)')
axes[0, 2].set_ylabel('Y-axis (BEV)')
axes[0, 2].set_xlabel('X-axis (BEV)')
axes[0, 2].legend()


# --- Prediction Visualization ---

# 4. Predicted Trajectories for Other Agents
# Take the 0th batch item
predicted_trajectories = prediction_outputs['predicted_trajectories'] # (N_agents, num_modes, num_future_steps, trajectory_point_dim)
mode_probabilities = prediction_outputs['mode_probabilities'] # (N_agents, num_modes)

axes[1, 0].imshow(semantic_seg_map, cmap='Paired', origin='lower', alpha=0.5) # Background
axes[1, 0].set_title('Prediction: Multi-modal Agent Trajectories')
axes[1, 0].set_ylabel('Y-axis (BEV)')
axes[1, 0].set_xlabel('X-axis (BEV)')

# The input `detected_agents_states_seq` has normalized coordinates.
# We need to convert them back to pixel coordinates for plotting.
# Assuming BEV space is from [-1, 1] mapped to [0, BEV_DIM-1]

def normalize_to_pixel(coord_norm, dim_size):
    return (coord_norm + 1) / 2 * (dim_size - 1)

current_agent_states = dummy_detected_agents_states_seq[0, -1, :, :].cpu().numpy() # Current frame, 0th batch item

for agent_idx in range(params['max_num_agents']):
    agent_x_current_norm = current_agent_states[agent_idx, 0]
    agent_y_current_norm = current_agent_states[agent_idx, 1]

    agent_x_current_pixel = normalize_to_pixel(agent_x_current_norm, params['bev_w'])
    agent_y_current_pixel = normalize_to_pixel(agent_y_current_norm, params['bev_h'])

    # Plot current agent position
    axes[1, 0].plot(agent_x_current_pixel, agent_y_current_pixel, 's', color='black', markersize=8, label=f'Agent {agent_idx} Current' if agent_idx == 0 else "")

    # Plot predicted trajectories for this agent
    for mode_idx in range(params['num_modes']):
        traj_mode = predicted_trajectories[agent_idx, mode_idx, :, :]
        prob = mode_probabilities[agent_idx, mode_idx]

        # Convert normalized coordinates to pixel coordinates
        traj_x_pixel = normalize_to_pixel(traj_mode[:, 0], params['bev_w'])
        traj_y_pixel = normalize_to_pixel(traj_mode[:, 1], params['bev_h'])

        # Use alpha based on mode probability
        axes[1, 0].plot(traj_x_pixel, traj_y_pixel, color=plt.cm.jet(prob), linestyle='--', alpha=prob, linewidth=2)
axes[1, 0].legend(['Agent Current Position', 'Predicted Trajectories (opacity by prob)'], loc='upper left')
axes[1, 0].set_xlim(0, params['bev_w']-1)
axes[1, 0].set_ylim(0, params['bev_h']-1)


# --- Planning Visualization ---

# 5. Selected Ego-Vehicle Trajectory
selected_trajectory = planning_outputs['selected_trajectory'] # (num_future_steps_plan, trajectory_point_dim_plan)

ego_x_current = normalize_to_pixel(dummy_ego_vehicle_state[0, 0].cpu().numpy(), params['bev_w']) # Assuming ego state has current x,y
ego_y_current = normalize_to_pixel(dummy_ego_vehicle_state[0, 1].cpu().numpy(), params['bev_h'])

# Convert normalized trajectory points to pixel coordinates
selected_traj_x_pixel = normalize_to_pixel(selected_trajectory[:, 0], params['bev_w'])
selected_traj_y_pixel = normalize_to_pixel(selected_trajectory[:, 1], params['bev_h'])

axes[1, 1].imshow(semantic_seg_map, cmap='Paired', origin='lower', alpha=0.5) # Background
axes[1, 1].plot(ego_x_current, ego_y_current, 'o', color='green', markersize=10, label='Ego Current Position')
axes[1, 1].plot(selected_traj_x_pixel, selected_traj_y_pixel, '-', color='blue', linewidth=3, label='Selected Ego Trajectory')

# Optionally plot other agents on this map for context
for agent_idx in range(params['max_num_agents']):
    agent_x_current_norm = current_agent_states[agent_idx, 0]
    agent_y_current_norm = current_agent_states[agent_idx, 1]
    agent_x_current_pixel = normalize_to_pixel(agent_x_current_norm, params['bev_w'])
    agent_y_current_pixel = normalize_to_pixel(agent_y_current_norm, params['bev_h'])
    axes[1, 1].plot(agent_x_current_pixel, agent_y_current_pixel, 's', color='gray', markersize=5)

axes[1, 1].set_title('Planning: Selected Ego Trajectory')
axes[1, 1].set_ylabel('Y-axis (BEV)')
axes[1, 1].set_xlabel('X-axis (BEV)')
axes[1, 1].legend(loc='upper left')
axes[1, 1].set_xlim(0, params['bev_w']-1)
axes[1, 1].set_ylim(0, params['bev_h']-1)

# 6. Occupancy Probabilities (a slice of it)
# Occupancy probabilities are (N_query, 1). We need to map them back to a grid for visualization.
# For simplicity, let's just show a scatter plot of occupied points or a 2D projection
occupancy_probs = perception_outputs['occupancy_probabilities'].squeeze()
occupancy_query_points_data = dummy_occupancy_query_points[0].cpu().numpy() # (N_query, 3)

# Filter points with high occupancy probability
occupied_indices = np.where(occupancy_probs > 0.5)[0]
if len(occupied_indices) > 0:
    occupied_points_norm = occupancy_query_points_data[occupied_indices, :2] # x,y
    occupied_x_pixel = normalize_to_pixel(occupied_points_norm[:, 0], params['bev_w'])
    occupied_y_pixel = normalize_to_pixel(occupied_points_norm[:, 1], params['bev_h'])
    axes[1, 2].scatter(occupied_x_pixel, occupied_y_pixel, color='purple', s=5, alpha=0.5)
axes[1, 2].imshow(semantic_seg_map, cmap='Paired', origin='lower', alpha=0.3) # Background
axes[1, 2].set_title('Perception: Occupancy Map (BEV)')
axes[1, 2].set_ylabel('Y-axis (BEV)')
axes[1, 2].set_xlabel('X-axis (BEV)')
axes[1, 2].set_xlim(0, params['bev_w']-1)
axes[1, 2].set_ylim(0, params['bev_h']-1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()