import torch
from tqdm.auto import tqdm

# --- Generic validate_epoch function ---
def validate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: callable, # Can be a single loss function or a dictionary of callables
    device: torch.device,
    task_type: str # 'perception', 'prediction', 'planning', or 'e2e'
) -> dict:
    model.eval() # Set model to evaluation mode
    model.to(device)
    total_loss_sum = 0.0 # Accumulator for total scalar loss for the epoch
    num_batches = 0

    # Dictionary to accumulate individual loss components for logging
    accumulated_individual_losses = {} # Will be populated based on keys from loss_function output

    with torch.no_grad(): # Disable gradient computation
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Validation {task_type} Epoch")):
            # Move inputs and ground truths to device
            inputs = {} # To hold all inputs for the model
            gts = {}    # To hold all ground truths for loss calculation

            # Dynamically prepare inputs and ground truths based on task_type
            if task_type == 'perception':
                inputs['camera_input'] = data['camera_input'].to(device)
                inputs['lidar_input'] = data['lidar_input'].to(device)
                inputs['radar_input'] = data['radar_input'].to(device)

                gts['gt_3d_boxes'] = data['gt_3d_boxes'].to(device)
                gts['gt_semantic_map'] = data['gt_semantic_map'].to(device)
                gts['gt_lane_boundaries'] = data['gt_lane_boundaries'].to(device)

            elif task_type == 'prediction':
                inputs['perceived_agent_current_states'] = data['perceived_agent_current_states'].to(device)
                inputs['historical_agent_trajectories'] = data['historical_agent_trajectories'].to(device)
                inputs['bev_semantic_map'] = data['bev_semantic_map'].to(device)
                inputs['ego_vehicle_state_features'] = data['ego_vehicle_state_features'].to(device)

                gts['gt_future_trajectories'] = data['gt_future_trajectories'].to(device)
                gts['gt_agent_intentions'] = data['gt_agent_intentions'].to(device)

            elif task_type == 'planning':
                inputs['bev_semantic_map'] = data['bev_semantic_map'].to(device)
                inputs['predicted_trajectories'] = data['predicted_trajectories'].to(device)
                inputs['trajectory_confidences'] = data['trajectory_confidences'].to(device)
                inputs['predicted_intentions'] = data['predicted_intentions'].to(device)
                inputs['ego_vehicle_state'] = data['ego_vehicle_state'].to(device)

                gts['gt_optimal_trajectory'] = data['gt_optimal_trajectory'].to(device)
                gts['gt_control_commands'] = data['gt_control_commands'].to(device)

            elif task_type == 'e2e':
                inputs['camera_input'] = data['camera_input'].to(device)
                inputs['lidar_input'] = data['lidar_input'].to(device)
                inputs['radar_input'] = data['radar_input'].to(device)
                inputs['ego_vehicle_state_features'] = data['ego_vehicle_state_features'].to(device) # ADDED THIS LINE

                gts['gt_3d_boxes'] = data['gt_3d_boxes'].to(device)
                gts['gt_semantic_map'] = data['gt_semantic_map'].to(device)
                gts['gt_lane_boundaries'] = data['gt_lane_boundaries'].to(device)
                gts['gt_future_trajectories'] = data['gt_future_trajectories'].to(device)
                gts['gt_agent_intentions'] = data['gt_agent_intentions'].to(device)
                gts['gt_optimal_trajectory'] = data['gt_optimal_trajectory'].to(device)
                gts['gt_control_commands'] = data['gt_control_commands'].to(device)
            else:
                raise ValueError(f"Unknown task_type for validation: {task_type}")

            # Forward pass
            outputs = model(**inputs)

            # Loss calculation
            loss_output = loss_function(outputs, gts)

            loss_to_accumulate = None
            if isinstance(loss_output, dict):
                # If the loss_function returned a dictionary (e.g., from PerceptionLoss),
                # extract the 'total_loss' for accumulation.
                loss_to_accumulate = loss_output['total_loss']
                for k, v in loss_output.items():
                    if k != 'total_loss':
                        if k not in accumulated_individual_losses:
                            accumulated_individual_losses[k] = 0.0
                        accumulated_individual_losses[k] += v.item()
            else:
                # If the loss_function returned a single tensor
                loss_to_accumulate = loss_output
                if 'total_loss' not in accumulated_individual_losses:
                    accumulated_individual_losses['total_loss'] = 0.0
                accumulated_individual_losses['total_loss'] += loss_to_accumulate.item()

            total_loss_sum += loss_to_accumulate.item()
            num_batches += 1

    avg_loss = total_loss_sum / num_batches
    results = {'total_loss': avg_loss}

    for name, value in accumulated_individual_losses.items():
        results[name] = value / num_batches

    return results

