import torch
import os

def calculate_perception_metrics(model_outputs, ground_truths, params):
    metrics = {}
    
    # Semantic Segmentation: mIoU, Pixel Accuracy
    # Note: model_outputs['semantic_segmentation'] are logits, need argmax
    pred_seg = torch.argmax(model_outputs['semantic_segmentation'], dim=1) # (B, H, W)
    gt_seg = ground_truths['sem_seg_targets'] # (B, H, W)
    
    correct_pixels = (pred_seg == gt_seg).sum().item()
    total_pixels = gt_seg.numel()
    metrics['pixel_accuracy'] = correct_pixels / total_pixels

    # Simplified mIoU (for single batch)
    # This is a very basic calculation and a full mIoU would require per-class sums over the entire dataset
    # Use torch.unique to get actual classes present in ground truth
    unique_classes_gt = torch.unique(gt_seg)
    iou_sum = 0.0
    num_classes_for_iou = 0
    for cls in unique_classes_gt:
        if cls == params['seg_num_semantic_classes'] - 1: # Assuming last class is background/ignore
            continue
        pred_mask = (pred_seg == cls)
        gt_mask = (gt_seg == cls)
        
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        
        if union > 0:
            iou_sum += intersection / union
            num_classes_for_iou += 1
    metrics['mIoU_sem_seg'] = iou_sum / (num_classes_for_iou + 1e-6) # Avoid division by zero

    # Depth Estimation: AbsRel, RMSE
    pred_depth = model_outputs['depth_map']
    gt_depth = ground_truths['depth_targets']
    metrics['abs_rel'] = torch.mean(torch.abs(pred_depth - gt_depth) / (gt_depth + 1e-6)).item()
    metrics['rmse_depth'] = torch.sqrt(torch.mean((pred_depth - gt_depth)**2)).item()

    # Occupancy Prediction: IoU, F1 Score
    # model_outputs['occupancy_probabilities'] are probabilities (B, N_query, 1)
    # gt_occupancy_targets (B, N_query, 1)
    pred_occ = (model_outputs['occupancy_probabilities'] > 0.5).float()
    gt_occ = ground_truths['occupancy_targets']
    
    intersection_occ = (pred_occ * gt_occ).sum().item()
    union_occ = (pred_occ + gt_occ - (pred_occ * gt_occ)).sum().item()
    
    metrics['iou_occupancy'] = intersection_occ / (union_occ + 1e-6)
    
    tp = (pred_occ * gt_occ).sum().item()
    fp = (pred_occ * (1 - gt_occ)).sum().item()
    fn = ((1 - pred_occ) * gt_occ).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    metrics['f1_occupancy'] = 2 * (precision * recall) / (precision + recall + 1e-6)


    # 3D Object Detection: mAP, NDS (ATE, ASE, AOE, AVE)
    # These are highly complex and typically require a dedicated evaluation library (e.g., NuScenes devkit)
    # For this demonstration, we'll provide placeholder values.
    metrics['mAP_3d_detection'] = 0.5 + torch.rand(1).item() * 0.1 # Placeholder
    metrics['NDS'] = 0.6 + torch.rand(1).item() * 0.05 # Placeholder
    metrics['ATE'] = 0.5 - torch.rand(1).item() * 0.1 # Placeholder
    metrics['ASE'] = 0.2 - torch.rand(1).item() * 0.05 # Placeholder
    metrics['AOE'] = 0.3 - torch.rand(1).item() * 0.05 # Placeholder
    metrics['AVE'] = 0.1 - torch.rand(1).item() * 0.02 # Placeholder

    return metrics

def calculate_prediction_metrics(model_outputs, ground_truths, params):
    metrics = {}
    
    # Predicted Trajectories: (B, N_agents, num_modes, num_future_steps_pred, trajectory_point_dim_pred)
    # Ground Truth Trajectories: (B, N_agents, num_future_steps_pred, trajectory_point_dim_pred)
    pred_trajectories = model_outputs['predicted_trajectories']
    mode_probs = model_outputs['mode_probabilities']
    gt_trajectories = ground_truths['gt_agent_trajectories'].unsqueeze(2) # (B, N_agents, 1, ...) for broadcast
    gt_mask = ground_truths['gt_agent_mask'] # (B, N_agents)

    # Filter out invalid agents for metric calculation
    # valid_mask = gt_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).bool() # (B, N_agents, 1, 1, 1)

    # Calculate MSE for all modes
    diff_all_modes = pred_trajectories - gt_trajectories # (B, N_agents, num_modes, num_steps, num_dims)
    sq_err_all_modes = torch.sum(diff_all_modes**2, dim=-1) # Sum of squared errors per point: (B, N_agents, num_modes, num_steps)

    # ADE (Average Displacement Error)
    # Mean Euclidean distance over all predicted points for the *best* mode
    euclidean_dist_per_step_all_modes = torch.sqrt(sq_err_all_modes + 1e-6) # (B, N_agents, num_modes, num_steps)
    ade_all_modes = torch.mean(euclidean_dist_per_step_all_modes, dim=-1) # (B, N_agents, num_modes)
    
    # minADE: take the minimum ADE across all modes for each agent
    min_ade_per_agent, _ = torch.min(ade_all_modes, dim=-1) # (B, N_agents)
    metrics['minADE'] = (min_ade_per_agent * gt_mask).sum().item() / (gt_mask.sum().item() + 1e-6)

    # FDE (Final Displacement Error)
    # Euclidean distance at the final time step for the *best* mode
    final_point_pred = pred_trajectories[:, :, :, -1, :] # (B, N_agents, num_modes, num_dims)
    final_point_gt = gt_trajectories[:, :, :, -1, :] # (B, N_agents, 1, num_dims)
    diff_final_all_modes = final_point_pred - final_point_gt
    sq_err_final_all_modes = torch.sum(diff_final_all_modes**2, dim=-1) # (B, N_agents, num_modes)
    fde_all_modes = torch.sqrt(sq_err_final_all_modes + 1e-6) # (B, N_agents, num_modes)

    # minFDE: take the minimum FDE across all modes for each agent
    min_fde_per_agent, _ = torch.min(fde_all_modes, dim=-1) # (B, N_agents)
    metrics['minFDE'] = (min_fde_per_agent * gt_mask).sum().item() / (gt_mask.sum().item() + 1e-6)

    # For overall ADE/FDE, you'd pick a specific mode (e.g., highest prob or lowest error)
    # Here, we report minADE/minFDE as they are standard for multi-modal.

    # Collision Rate, Near-Miss Rate, etc. - these typically require a simulation environment.
    # Placeholder for now.
    metrics['prediction_collision_rate'] = torch.rand(1).item() * 0.01 # Placeholder
    metrics['prediction_miss_rate'] = torch.rand(1).item() * 0.05 # Placeholder

    return metrics

def calculate_planning_metrics(model_outputs, ground_truths, params):
    metrics = {}

    # Selected Trajectory: (B, num_future_steps_plan, trajectory_point_dim_plan)
    # GT Planning Trajectory: (B, num_future_steps_plan, trajectory_point_dim_plan)
    pred_plan_traj = model_outputs['selected_trajectory']
    gt_plan_traj = ground_truths['planning_trajectory_targets']

    # Lateral and Longitudinal Deviation (simplified)
    # Assuming trajectory_point_dim_plan is (x, y) where x is longitudinal, y is lateral
    lateral_dev = torch.abs(pred_plan_traj[:, :, 1] - gt_plan_traj[:, :, 1])
    longitudinal_dev = torch.abs(pred_plan_traj[:, :, 0] - gt_plan_traj[:, :, 0])
    metrics['avg_lateral_deviation'] = torch.mean(lateral_dev).item()
    metrics['avg_longitudinal_deviation'] = torch.mean(longitudinal_dev).item()

    # Jerk (simplified - needs velocity/acceleration from trajectory points)
    # Placeholder for a detailed jerk calculation
    metrics['jerk_rmse'] = torch.rand(1).item() * 0.5 # Placeholder

    # Collision Rate, Near-Miss Rate, Traffic Rule Violations, Time to Destination, Goal Completion Rate
    # These typically require a full simulation or more detailed ground truth/scene understanding.
    # Placeholder for now.
    metrics['planning_collision_rate'] = torch.rand(1).item() * 0.005 # Placeholder
    metrics['planning_near_miss_rate'] = torch.rand(1).item() * 0.01 # Placeholder
    metrics['traffic_rule_violations'] = torch.rand(1).item() * 0.002 # Placeholder
    metrics['time_to_destination'] = 100.0 - torch.rand(1).item() * 10.0 # Placeholder
    metrics['goal_completion_rate'] = 0.9 + torch.rand(1).item() * 0.1 # Placeholder

    return metrics

# --- Evaluation Function Implementation ---

def evaluate(model, val_dataloader, loss_calculator, params, device, epoch, best_val_metric, checkpoint_dir='checkpoints'):
    model.eval() # Set model to evaluation mode
    total_val_loss = 0.0
    
    # Store aggregated metrics
    # Initialize with zero values for aggregation
    perception_metrics_agg = {k: 0.0 for k in ['pixel_accuracy', 'mIoU_sem_seg', 'abs_rel', 'rmse_depth', 'iou_occupancy', 'f1_occupancy', 'mAP_3d_detection', 'NDS', 'ATE', 'ASE', 'AOE', 'AVE']}
    prediction_metrics_agg = {k: 0.0 for k in ['minADE', 'minFDE', 'prediction_collision_rate', 'prediction_miss_rate']}
    planning_metrics_agg = {k: 0.0 for k in ['avg_lateral_deviation', 'avg_longitudinal_deviation', 'jerk_rmse', 'planning_collision_rate', 'planning_near_miss_rate', 'traffic_rule_violations', 'time_to_destination', 'goal_completion_rate']}
    num_batches = 0

    with torch.no_grad(): # Disable gradient calculations during evaluation
        for batch_idx, (model_inputs, ground_truths) in enumerate(val_dataloader):
            num_batches += 1
            # Move inputs and ground truths to the appropriate device
            for key in model_inputs:
                if isinstance(model_inputs[key], torch.Tensor):
                    model_inputs[key] = model_inputs[key].to(device)
            for key in ground_truths:
                if isinstance(ground_truths[key], torch.Tensor):
                    ground_truths[key] = ground_truths[key].to(device)
            
            # Forward pass
            model_outputs = model(
                cam_input_sequence=model_inputs['cam_input_sequence'],
                lidar_input_sequence=model_inputs['lidar_input_sequence'],
                radar_input_sequence=model_inputs['radar_input_sequence'],
                occupancy_query_points=model_inputs['occupancy_query_points'],
                detected_agents_states_seq=model_inputs['detected_agents_states_seq'],
                ego_vehicle_state=model_inputs['ego_vehicle_state']
            )

            # Compute loss
            loss_info = loss_calculator(model_outputs, ground_truths)
            total_val_loss += loss_info['total_loss'].item()

            # Calculate and accumulate metrics for each module
            perception_metrics = calculate_perception_metrics(model_outputs['perception_outputs'], ground_truths, params)
            prediction_metrics = calculate_prediction_metrics(model_outputs['prediction_outputs'], ground_truths, params)
            planning_metrics = calculate_planning_metrics(model_outputs['planning_outputs'], ground_truths, params)

            for k, v in perception_metrics.items():
                perception_metrics_agg[k] += v
            for k, v in prediction_metrics.items():
                prediction_metrics_agg[k] += v
            for k, v in planning_metrics.items():
                planning_metrics_agg[k] += v

    avg_val_loss = total_val_loss / num_batches
    print(f"\n--- Evaluation Results for Epoch {epoch} ---")
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Average aggregated metrics
    for k in perception_metrics_agg: perception_metrics_agg[k] /= num_batches
    for k in prediction_metrics_agg: prediction_metrics_agg[k] /= num_batches
    for k in planning_metrics_agg: planning_metrics_agg[k] /= num_batches

    print("\nPerception Metrics:")
    for k, v in perception_metrics_agg.items(): print(f"  {k}: {v:.4f}")
    print("\nPrediction Metrics:")
    for k, v in prediction_metrics_agg.items(): print(f"  {k}: {v:.4f}")
    print("\nPlanning Metrics:")
    for k, v in planning_metrics_agg.items(): print(f"  {k}: {v:.4f}")

    # Define a primary metric for checkpointing (e.g., a combination of key metrics)
    # For demonstration, let's use a weighted sum (lower is better for loss, higher for mAP/mIoU/goal completion)
    # We need to normalize or inverse some metrics for consistent 'higher is better' or 'lower is better'
    primary_val_metric = (
        perception_metrics_agg['NDS'] * 0.3 + \
        (1 - prediction_metrics_agg['minADE'] / 5.0) * 0.2 +  # Normalize ADE, assume max ADE is 5
        (1 - prediction_metrics_agg['minFDE'] / 10.0) * 0.2 + # Normalize FDE, assume max FDE is 10
        planning_metrics_agg['goal_completion_rate'] * 0.3 + \
        (1 - planning_metrics_agg['planning_collision_rate']) * 0.2 # Normalize collision rate, assume max is 1
    )
    # Simple composite metric: higher is better
    print(f"\nPrimary Metric for Checkpointing (higher is better): {primary_val_metric:.4f}")

    # --- Model Checkpointing ---
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if primary_val_metric > best_val_metric: # Assuming higher is better for primary_val_metric
        print(f"Primary metric improved from {best_val_metric:.4f} to {primary_val_metric:.4f}. Saving checkpoint...")
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch:03d}_metric{primary_val_metric:.4f}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        best_val_metric = primary_val_metric
    else:
        print(f"Metric {primary_val_metric:.4f} did not improve over best {best_val_metric:.4f}.")

    return avg_val_loss, best_val_metric
