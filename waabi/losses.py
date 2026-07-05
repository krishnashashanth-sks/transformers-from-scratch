import torch.nn.functional as F
import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return focal_loss.mean()

class CrossEntropyLossSeg(nn.Module):
    def __init__(self):
        super(CrossEntropyLossSeg, self).__init__()

    def forward(self, inputs, targets):
        # For semantic segmentation, targets are usually long-type pixel labels
        # inputs are raw logits (batch, num_classes, H, W)
        # targets are (batch, H, W)
        # If targets are float (e.g., one-hot encoded or probabilities), they need to be converted to LongTensor
        # However, for nn.CrossEntropyLoss, targets should be class indices (LongTensor)
        # Ensure targets are of type long and in the correct shape (B, H, W)
        # Also, inputs are typically logits, so no softmax needed here
        return F.cross_entropy(inputs, targets.long())
    
class IntentionLoss(nn.Module):
    def __init__(self):
        super(IntentionLoss, self).__init__()

    def forward(self, predicted_intentions, gt_agent_intentions):
        # predicted_intentions: (B, N_agents, num_intention_classes) - logits, before softmax
        # gt_agent_intentions: (B, N_agents) - class indices

        # Reshape for F.cross_entropy, which expects (N, C) for inputs and (N) for targets
        batch_size, num_agents, num_classes = predicted_intentions.shape
        predicted_intentions_reshaped = predicted_intentions.view(batch_size * num_agents, num_classes)
        gt_agent_intentions_reshaped = gt_agent_intentions.view(batch_size * num_agents)

        return F.cross_entropy(predicted_intentions_reshaped, gt_agent_intentions_reshaped)

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    def forward(self, pred, target):
        if target.dtype != pred.dtype:
            target = target.to(pred.dtype)
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        return loss.mean()

class TrajectoryLoss(nn.Module):
    def __init__(self, pose_dim=3, beta=1.0):
        super(TrajectoryLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss(beta=beta)
        self.pose_dim = pose_dim

    def forward(self, predicted_trajectories, trajectory_confidences, gt_future_trajectories):
        # predicted_trajectories: (B, N_agents, K, T, pose_dim)
        # trajectory_confidences: (B, N_agents, K) - logits, before softmax
        # gt_future_trajectories: (B, N_agents, T, pose_dim)

        batch_size, num_agents, num_output_trajectories, prediction_horizon, _ = predicted_trajectories.shape

        # Reshape for easier calculation (B*N_agents, K, T*pose_dim)
        predicted_trajectories_flat = predicted_trajectories.view(batch_size * num_agents, num_output_trajectories, -1)
        gt_future_trajectories_flat = gt_future_trajectories.view(batch_size * num_agents, -1)

        # Calculate SmoothL1Loss for each of the K predicted trajectories against GT
        # Output will be (B*N_agents, K)
        regression_losses = torch.zeros(batch_size * num_agents, num_output_trajectories, device=predicted_trajectories.device)
        for k in range(num_output_trajectories):
            regression_losses[:, k] = self.smooth_l1_loss(predicted_trajectories_flat[:, k, :], gt_future_trajectories_flat)

        # Find the best_k index for each agent (one that minimizes the regression loss)
        min_regression_losses, best_k_indices = torch.min(regression_losses, dim=1)

        # Trajectory Regression Loss: use the minimum regression loss for each agent
        trajectory_reg_loss = min_regression_losses.mean()

        # Confidence Classification Loss: encourage high confidence for the best_k trajectory
        # Reshape confidences to (B*N_agents, K) for F.cross_entropy
        trajectory_confidences_reshaped = trajectory_confidences.view(batch_size * num_agents, num_output_trajectories)
        # Cross-entropy loss expects logits for inputs and class indices for targets
        confidence_cls_loss = F.cross_entropy(trajectory_confidences_reshaped, best_k_indices)

        # Combine the two components
        total_trajectory_loss = trajectory_reg_loss + confidence_cls_loss

        return total_trajectory_loss, trajectory_reg_loss, confidence_cls_loss

class PerceptionLoss(nn.Module):
    def __init__(self, loss_functions: dict, loss_weights: dict):
        super(PerceptionLoss, self).__init__()
        self.loss_functions = nn.ModuleDict({k: v for k, v in loss_functions.items()})
        self.loss_weights = loss_weights

    def forward(self, predictions: dict, ground_truths: dict):
        total_loss = 0.0
        # Handle bbox_cls_loss and bbox_reg_loss together for 3D boxes
        # Assuming predictions['3d_boxes'] has a structure that can be split into class and regression targets
        # and ground_truths['gt_3d_boxes'] has corresponding structure.
        # For simplicity, we'll assume predictions are directly comparable to GT for now.

        # Example of how bbox_cls_loss and bbox_reg_loss might be computed:
        num_predicted_boxes = predictions['3d_boxes'].shape[1] # e.g., 10
        num_gt_boxes_attrs = ground_truths['gt_3d_boxes'].shape[2] # e.g., 13 (10 reg + 3 classes)
        num_classes_boxes_in_gt = 3 # Hardcoded based on CustomDataset
        num_reg_attrs_in_gt = num_gt_boxes_attrs - num_classes_boxes_in_gt # 10

        # For bbox_cls_loss:
        # `PerceptionNet`'s bbox_head output needs to be split. Let's assume the last 3 elements of each object's 12 attributes are class logits.
        # So prediction `(batch, 10, 12)` -> regression `(batch, 10, 9)` and classification `(batch, 10, 3)`
        # `ground_truths['gt_3d_boxes']` `(batch, 10, 13)` -> regression `(batch, 10, 10)` and classification `(batch, 10, 3)`

        # Let's simplify and assume for now the PerceptionNet `bbox_predictions` already aligns with regression and classification parts
        # For bbox_cls_loss, we expect a logit output for each box and the GT class (one-hot or index)
        # For perception_net, bbox_predictions: (batch_size, 10, 9 + num_classes_boxes) -> (batch_size, 10, 12)
        # We'll assume the last 3 values are class logits for 3 classes.
        predicted_bbox_cls_logits = predictions['3d_boxes'][:, :, -num_classes_boxes_in_gt:] # Assuming last 3 for classes
        gt_bbox_classes_one_hot = ground_truths['gt_3d_boxes'][:, :, -num_classes_boxes_in_gt:] # Assuming last 3 for classes

        # Clamp values to avoid issues with FocalLoss input range if it's expecting probabilities
        # Or ensure FocalLoss is used with `F.binary_cross_entropy_with_logits` internally as done in its definition
        bbox_cls_loss = self.loss_functions['bbox_cls_loss'](predicted_bbox_cls_logits, gt_bbox_classes_one_hot)
        total_loss += self.loss_weights['bbox_cls_loss'] * bbox_cls_loss

        # Bounding Box Regression Loss (SmoothL1Loss)
        predicted_bbox_reg = predictions['3d_boxes'][:, :, :num_reg_attrs_in_gt] # Assuming first 10 for regression (x,y,z,yaw,l,w,h,vx,vy,vz)
        gt_bbox_reg = ground_truths['gt_3d_boxes'][:, :, :num_reg_attrs_in_gt] # Assuming first 10 for regression
        bbox_reg_loss = self.loss_functions['bbox_reg_loss'](predicted_bbox_reg, gt_bbox_reg)
        total_loss += self.loss_weights['bbox_reg_loss'] * bbox_reg_loss

        # Semantic Segmentation Loss (CrossEntropyLossSeg)
        semantic_loss = self.loss_functions['semantic_loss'](predictions['semantic_map'], ground_truths['gt_semantic_map'])
        total_loss += self.loss_weights['semantic_loss'] * semantic_loss

        # Lane Boundary Detection Loss (SmoothL1Loss)
        lane_loss = self.loss_functions['lane_loss'](predictions['lane_boundaries'], ground_truths['gt_lane_boundaries'])
        total_loss += self.loss_weights['lane_loss'] * lane_loss

        # Return individual losses as well for logging
        return {
            'total_loss': total_loss,
            'bbox_cls_loss': bbox_cls_loss,
            'bbox_reg_loss': bbox_reg_loss,
            'semantic_loss': semantic_loss,
            'lane_loss': lane_loss
        }
    
class PlanningLoss(nn.Module):
    def __init__(self, loss_weights: dict, pose_dim=3, control_dim=3, beta=1.0):
        super(PlanningLoss, self).__init__()
        self.trajectory_loss_fn = SmoothL1Loss(beta=beta) # For optimal trajectory (sequence of poses)
        self.control_loss_fn = SmoothL1Loss(beta=beta)    # For control commands (steering, throttle, brake)
        self.loss_weights = loss_weights
        self.pose_dim = pose_dim
        self.control_dim = control_dim

    def forward(self, predictions: dict, ground_truths: dict):
        # predictions from PlanningNet
        #   'optimal_trajectory': (B, target_trajectory_len, pose_dim)
        #   'control_commands': (B, control_dim)

        # ground_truths from CustomDataset
        #   'gt_optimal_trajectory': (B, target_trajectory_len, pose_dim)
        #   'gt_control_commands': (B, control_dim)

        # --- Optimal Trajectory Loss (SmoothL1Loss) ---
        # Ensure ground truth is float for consistency
        gt_optimal_trajectory_float = ground_truths['gt_optimal_trajectory'].to(predictions['optimal_trajectory'].dtype)
        L_traj = self.trajectory_loss_fn(predictions['optimal_trajectory'], gt_optimal_trajectory_float)

        # --- Control Command Loss (SmoothL1Loss) ---
        # Ensure ground truth is float for consistency
        gt_control_commands_float = ground_truths['gt_control_commands'].to(predictions['control_commands'].dtype)
        L_control = self.control_loss_fn(predictions['control_commands'], gt_control_commands_float)

        # --- Total Planning Loss ---
        total_loss = (
            self.loss_weights.get('optimal_trajectory_loss', 1.0) * L_traj +
            self.loss_weights.get('control_command_loss', 1.0) * L_control
        )

        # Optional: Safety/Comfort Loss could be added here if defined and implemented.
        # For now, we omit them as per the conceptual implementation.

        return {
            'total_loss': total_loss,
            'optimal_trajectory_loss': L_traj,
            'control_command_loss': L_control
        }

class PredictionLoss(nn.Module):
    def __init__(self, loss_weights: dict, pose_dim=3, beta=1.0, num_agents_for_dataset=5):
        super(PredictionLoss, self).__init__()
        self.trajectory_loss_fn = TrajectoryLoss(pose_dim=pose_dim, beta=beta)
        self.intention_loss_fn = IntentionLoss()
        self.loss_weights = loss_weights
        self.num_agents_for_dataset = num_agents_for_dataset # Store this for potential use in handling outputs

    def forward(self, predictions: dict, ground_truths: dict):
        # predictions from PredictionNet
        #   'predicted_trajectories': (B, N_agents, K, T, pose_dim)
        #   'trajectory_confidences': (B, N_agents, K) - logits
        #   'predicted_intentions': (B, N_agents, num_intention_classes) - logits

        # ground_truths from CustomDataset
        #   'gt_future_trajectories': (B, N_agents, T, pose_dim)
        #   'gt_agent_intentions': (B, N_agents) - class indices

        batch_size = predictions['predicted_trajectories'].shape[0]
        # The CustomDataset generates data for a fixed number of agents (num_agents_for_dataset)
        # But the batch could have fewer valid agents if some were filtered out.
        # For now, we assume all agents in the batch are valid and align with the first num_agents_for_dataset entries.
        # Filter out agents that are just zeros (padding from CustomDataset if actual agents were fewer)
        # A more robust solution would pass the actual number of valid agents in the batch.

        # Filter out padding for predictions
        # Assuming predictions are padded with zeros if there are fewer actual agents than `num_agents_for_dataset`
        # This is a simplification; ideally, models handle variable number of agents directly or use attention masks
        valid_agent_mask = (ground_truths['gt_agent_intentions'] != 0).any(dim=1) # Example mask: if any intention is not 0 (e.g., straight is 0)
        # This mask may not be perfect if 0 is a valid intention.
        # A better mask might come from the perception module indicating valid agents.
        # For now, rely on `num_agents` dimension from predictions which should align with GT.

        # --- Trajectory Loss ---
        L_traj_total, L_traj_reg, L_traj_conf = self.trajectory_loss_fn(
            predictions['predicted_trajectories'],
            predictions['trajectory_confidences'],
            ground_truths['gt_future_trajectories']
        )

        # --- Intention Loss ---
        L_intent = self.intention_loss_fn(
            predictions['predicted_intentions'],
            ground_truths['gt_agent_intentions']
        )

        # --- Total Prediction Loss ---
        total_loss = (
            self.loss_weights.get('trajectory_loss', 1.0) * L_traj_total +
            self.loss_weights.get('intention_loss', 1.0) * L_intent
        )

        return {
            'total_loss': total_loss,
            'trajectory_loss': L_traj_total,
            'trajectory_reg_loss': L_traj_reg,
            'trajectory_conf_loss': L_traj_conf,
            'intention_loss': L_intent
        }

class EndToEndLoss(nn.Module):
    def __init__(self, perception_loss_fn, prediction_loss_fn, planning_loss_fn, e2e_loss_weights):
        super(EndToEndLoss, self).__init__()
        self.perception_loss_fn = perception_loss_fn
        self.prediction_loss_fn = prediction_loss_fn
        self.planning_loss_fn = planning_loss_fn
        self.e2e_loss_weights = e2e_loss_weights

    def forward(self, model_outputs: dict, ground_truths: dict):
        # Calculate perception losses
        perception_gt = {
            'gt_3d_boxes': ground_truths['gt_3d_boxes'],
            'gt_semantic_map': ground_truths['gt_semantic_map'],
            'gt_lane_boundaries': ground_truths['gt_lane_boundaries']
        }
        perception_losses = self.perception_loss_fn(model_outputs['perception_outputs'], perception_gt)

        # Calculate prediction losses
        prediction_gt = {
            'gt_future_trajectories': ground_truths['gt_future_trajectories'],
            'gt_agent_intentions': ground_truths['gt_agent_intentions']
        }
        prediction_losses = self.prediction_loss_fn(model_outputs['prediction_outputs'], prediction_gt)

        # Calculate planning losses
        planning_gt = {
            'gt_optimal_trajectory': ground_truths['gt_optimal_trajectory'],
            'gt_control_commands': ground_truths['gt_control_commands']
        }
        planning_losses = self.planning_loss_fn(model_outputs['planning_outputs'], planning_gt)

        # Combine all losses
        total_e2e_loss = (
            self.e2e_loss_weights.get('perception_loss', 1.0) * perception_losses['total_loss'] +
            self.e2e_loss_weights.get('prediction_loss', 1.0) * prediction_losses['total_loss'] +
            self.e2e_loss_weights.get('planning_loss', 1.0) * planning_losses['total_loss']
        )

        # Return a dictionary of all losses for logging
        combined_losses = {
            'total_loss': total_e2e_loss,
            'perception_total_loss': perception_losses['total_loss'],
            **{f'perception_{k}': v for k, v in perception_losses.items() if k != 'total_loss'},
            'prediction_total_loss': prediction_losses['total_loss'],
            **{f'prediction_{k}': v for k, v in prediction_losses.items() if k != 'total_loss'},
            'planning_total_loss': planning_losses['total_loss'],
            **{f'planning_{k}': v for k, v in planning_losses.items() if k != 'total_loss'},
        }

        return combined_losses
