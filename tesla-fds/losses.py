import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
  def __init__(self,gamma=2.0,alpha=0.25,reduction='mean',ignore_index=-100):
    super().__init__()
    self.gamma=gamma
    self.alpha=alpha
    self.reduction=reduction
    self.ignore_index=ignore_index
  def forward(self,inputs,targets):
    if self.ignore_index>=0:
      mask=(targets!=self.ignore_index).float()
      targets=targets.clone()
      targets[targets==self.ignore_index]=0
    ce_loss=F.cross_entropy(inputs,targets,reduction='none')
    pt=torch.exp(-ce_loss)
    focal_loss=self.alpha*(1-pt)**self.gamma*ce_loss
    if self.ignore_index>=0:
      focal_loss=focal_loss*mask
    if self.reduction=='mean':
      if self.ignore_index>=0:
        return focal_loss.sum()/mask.sum()
      else:
        return focal_loss.mean()
    elif self.reduction=='sum':
      return focal_loss.sum()
    else:
      return focal_loss
    
class GIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_reg_outputs, gt_boxes_per_pixel, gt_masks):
        """
        Calculates 3D GIoU loss for object detection bounding box regression.
        Input boxes are expected to be (cx, cy, cz, dx, dy, dz).
        Args:
            pred_reg_outputs (torch.Tensor): Predicted regression outputs from ObjectDetectionHead (B, N_reg, H, W).
                                             N_reg includes (cx, cy, cz, dx, dy, dz, yaw, vx, vy).
            gt_boxes_per_pixel (torch.Tensor): Ground truth boxes for each pixel (B, 6, H, W).
                                               Contains (cx, cy, cz, dx, dy, dz).
            gt_masks (torch.Tensor): Mask indicating where objects are present (B, 1, H, W).
        Returns:
            torch.Tensor: GIoU loss.
        """
        # Extract relevant predicted box parameters (cx, cy, cz, dx, dy, dz)
        # Assuming these are the first 6 parameters in pred_reg_outputs
        pred_boxes_raw = pred_reg_outputs[:, :6, :, :] # (B, 6, H, W)

        # Flatten for element-wise operations
        pred_boxes = pred_boxes_raw.permute(0, 2, 3, 1).reshape(-1, 6) # (B*H*W, 6)
        gt_boxes = gt_boxes_per_pixel.permute(0, 2, 3, 1).reshape(-1, 6) # (B*H*W, 6)
        mask = gt_masks.flatten() # (B*H*W)

        # Apply mask to consider only locations with ground truth objects
        pred_boxes_masked = pred_boxes[mask == 1]
        gt_boxes_masked = gt_boxes[mask == 1]

        if pred_boxes_masked.numel() == 0:
            # No ground truth boxes present, loss is 0
            return torch.tensor(0.0, device=pred_reg_outputs.device, requires_grad=True)

        # Extract box components
        pred_cx, pred_cy, pred_cz, pred_dx, pred_dy, pred_dz = pred_boxes_masked.split(1, dim=-1)
        gt_cx, gt_cy, gt_cz, gt_dx, gt_dy, gt_dz = gt_boxes_masked.split(1, dim=-1)

        # Ensure dimensions are positive (can be constrained in model output or clamped here)
        pred_dx = pred_dx.abs().clamp(min=1e-6)
        pred_dy = pred_dy.abs().clamp(min=1e-6)
        pred_dz = pred_dz.abs().clamp(min=1e-6)
        gt_dx = gt_dx.abs().clamp(min=1e-6)
        gt_dy = gt_dy.abs().clamp(min=1e-6)
        gt_dz = gt_dz.abs().clamp(min=1e-6)

        # Calculate coordinates of corners (min/max)
        pred_x1 = pred_cx - pred_dx / 2
        pred_y1 = pred_cy - pred_dy / 2
        pred_z1 = pred_cz - pred_dz / 2
        pred_x2 = pred_cx + pred_dx / 2
        pred_y2 = pred_cy + pred_dy / 2
        pred_z2 = pred_cz + pred_dz / 2

        gt_x1 = gt_cx - gt_dx / 2
        gt_y1 = gt_cy - gt_dy / 2
        gt_z1 = gt_cz - gt_dz / 2
        gt_x2 = gt_cx + gt_dx / 2
        gt_y2 = gt_cy + gt_dy / 2
        gt_z2 = gt_cz + gt_dz / 2

        # Calculate intersection coordinates
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_z1 = torch.max(pred_z1, gt_z1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        inter_z2 = torch.min(pred_z2, gt_z2)

        # Calculate intersection volume
        inter_dx = (inter_x2 - inter_x1).clamp(min=0)
        inter_dy = (inter_y2 - inter_y1).clamp(min=0)
        inter_dz = (inter_z2 - inter_z1).clamp(min=0)
        inter_volume = inter_dx * inter_dy * inter_dz

        # Calculate union volume
        pred_volume = pred_dx * pred_dy * pred_dz
        gt_volume = gt_dx * gt_dy * gt_dz
        union_volume = pred_volume + gt_volume - inter_volume

        union_volume = union_volume.clamp(min=1e-6)
        iou = inter_volume / union_volume

        # Calculate convex hull (smallest enclosing box) coordinates
        c_x1 = torch.min(pred_x1, gt_x1)
        c_y1 = torch.min(pred_y1, gt_y1)
        c_z1 = torch.min(pred_z1, gt_z1)
        c_x2 = torch.max(pred_x2, gt_x2)
        c_y2 = torch.max(pred_y2, gt_y2)
        c_z2 = torch.max(pred_z2, gt_z2)

        # Calculate convex hull volume
        c_dx = (c_x2 - c_x1).clamp(min=0)
        c_dy = (c_y2 - c_y1).clamp(min=0)
        c_dz = (c_z2 - c_z1).clamp(min=0)
        c_volume = c_dx * c_dy * c_dz

        c_volume = c_volume.clamp(min=1e-6)
        giou = iou - (c_volume - union_volume) / c_volume

        giou_loss = 1. - giou

        if self.reduction == 'mean':
            return giou_loss.mean()
        elif self.reduction == 'sum':
            return giou_loss.sum()
        else:
            return giou_loss
        
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Inputs: (N, C, H, W) - logits or probabilities
        # Targets: (N, H, W) - class indices for multi-class, or (N, 1, H, W) for binary

        if inputs.dim() == 4 and inputs.shape[1] > 1: # Multi-class segmentation
            # Convert logits to probabilities and then to one-hot for Dice
            inputs = F.softmax(inputs, dim=1) # (N, C, H, W)
            # Convert targets to one-hot encoding
            targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        elif inputs.dim() == 4 and inputs.shape[1] == 1: # Binary segmentation/occupancy with single channel logits
            inputs = torch.sigmoid(inputs) # (N, 1, H, W)
            targets_one_hot = targets.float() # Targets should be (N, 1, H, W) for binary
        else: # Handle other cases like flattened inputs if necessary
            raise ValueError("Unsupported input shape for DiceLoss")

        # Flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)

        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets_one_hot.sum() + self.smooth)

        loss = 1 - dice

        if self.reduction == 'mean':
            return loss # Since we flattened, 'mean' effectively means mean over batch and spatial dims
        elif self.reduction == 'sum':
            return loss * inputs.shape[0]
        else:
            return loss

class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Inputs: (N, 1, H, W) or (N, H, W) - logits or probabilities for binary classification
        # Targets: (N, 1, H, W) or (N, H, W) - binary ground truth (0 or 1)

        # Ensure inputs and targets are float and match dimensions (flattening if needed)
        if inputs.dim() > targets.dim():
            # If inputs is (N, C, H, W) and C is 1, squeeze C. If C > 1, this is an error.
            if inputs.shape[1] != 1:
                raise ValueError("BCELoss expects single channel input for binary classification.")
            inputs = inputs.squeeze(1)

        inputs = inputs.float()
        targets = targets.float()

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs and targets are float
        inputs = inputs.float()
        targets = targets.float()

        loss = F.l1_loss(inputs, targets, reduction='none')

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs and targets are float
        inputs = inputs.float()
        targets = targets.float()

        loss = F.mse_loss(inputs, targets, reduction='none')

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class NLLLossTrajectory(nn.Module):
    def __init__(self, reduction='mean', epsilon=1e-6, use_best_mode_nll=True, sigma=1.0):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon # Small value to prevent log(0)
        self.use_best_mode_nll = use_best_mode_nll
        self.sigma = sigma # A scalar for the Gaussian likelihood std, or a tensor if learned

    def forward(self, predicted_trajectories, mode_probabilities, gt_trajectories, gt_mask=None):
        """
        Calculates Negative Log-Likelihood Loss for multi-modal trajectory prediction.

        Args:
            predicted_trajectories (torch.Tensor): Predicted multi-modal trajectories.
                                                 Shape: (B, N_agents, num_modes, num_future_steps, trajectory_point_dim)
            mode_probabilities (torch.Tensor): Predicted probabilities for each mode.
                                            Shape: (B, N_agents, num_modes) (should sum to 1 along last dim)
            gt_trajectories (torch.Tensor): Ground truth trajectory.
                                          Shape: (B, N_agents, num_future_steps, trajectory_point_dim)
            gt_mask (torch.Tensor, optional): Mask indicating valid ground truth agents.
                                              Shape: (B, N_agents). Defaults to None (all agents are valid).

        Returns:
            torch.Tensor: NLL loss.
        """
        B, N_agents, num_modes, num_future_steps, trajectory_point_dim = predicted_trajectories.shape

        # Expand ground truth to compare with each mode
        gt_expanded = gt_trajectories.unsqueeze(2) # (B, N_agents, 1, num_future_steps, trajectory_point_dim)

        # Calculate squared Euclidean distance (MSE) between GT and each predicted mode
        # Sum over trajectory_point_dim and num_future_steps
        diff = predicted_trajectories - gt_expanded
        mse_per_mode = torch.mean(diff**2, dim=[-1, -2]) # (B, N_agents, num_modes)

        # Ensure mode_probabilities sum to 1 (or close to 1) for numerical stability
        # Softmax is already applied in TrajectoryDecoder, so we might just clamp or re-normalize if needed.
        # For safety, let's re-normalize if the sum is not close to 1. Epsilon helps prevent division by zero.
        mode_probabilities = mode_probabilities / (mode_probabilities.sum(dim=-1, keepdim=True) + self.epsilon)

        if self.use_best_mode_nll:
            # Approach 1: NLL based on the best mode (closest to GT), common for minADE/minFDE-like training
            # Find the mode that is closest to the ground truth (minimum MSE)
            min_mse_per_mode, best_mode_idx = torch.min(mse_per_mode, dim=-1) # (B, N_agents)

            # Gather the probability of the best mode
            best_mode_probabilities = torch.gather(mode_probabilities, dim=2, index=best_mode_idx.unsqueeze(-1)).squeeze(-1) # (B, N_agents)

            # NLL loss for this approach: -log(P_best) + 0.5 * min_mse / sigma^2
            # This combines a classification-like loss for mode selection with a regression loss for trajectory accuracy.
            # The 0.5 * min_mse / sigma^2 term acts as the negative log-likelihood of the trajectory point itself
            # assuming a Gaussian distribution with fixed variance sigma^2. Ignoring constant terms like log(2*pi*sigma^2).
            nll_loss_unreduced = -torch.log(best_mode_probabilities + self.epsilon) + 0.5 * min_mse_per_mode / (self.sigma**2 + self.epsilon)

        else:
            # Approach 2: Full NLL over the mixture distribution
            # log(P_k) is `torch.log(mode_probabilities + self.epsilon)`
            # log_likelihood_k_unscaled = -0.5 * mse_per_mode / sigma^2
            # log_mix_weights = log(P_k) + log_likelihood_k_unscaled
            log_mix_weights = torch.log(mode_probabilities + self.epsilon) - 0.5 * mse_per_mode / (self.sigma**2 + self.epsilon)

            # Use logsumexp for numerical stability to compute log(sum(exp(log_terms)))
            log_likelihood_agent = torch.logsumexp(log_mix_weights, dim=-1) # (B, N_agents)

            nll_loss_unreduced = -log_likelihood_agent # (B, N_agents)

        # Apply ground truth mask if provided (e.g., for agents that are actually present)
        if gt_mask is not None:
            # Ensure mask is broadcastable and matches N_agents dimension
            if gt_mask.dim() == 1: # (B,) mask, assuming N_agents is fixed to max_num_agents
                gt_mask = gt_mask.unsqueeze(1).expand(-1, N_agents)
            elif gt_mask.dim() == 3: # (B, T, N_agents), assuming we only care about current frame's mask
                gt_mask = gt_mask[:, -1, :]
            elif gt_mask.dim() != 2 or gt_mask.shape != (B, N_agents):
                raise ValueError(f"gt_mask has incompatible shape {gt_mask.shape}. Expected (B, N_agents).")

            nll_loss_unreduced = nll_loss_unreduced * gt_mask.float()
            valid_agents_count = gt_mask.sum()
        else:
            valid_agents_count = B * N_agents # All agents in batch are considered valid

        if valid_agents_count == 0:
            # If no valid agents, return a zero loss tensor
            return torch.tensor(0.0, device=predicted_trajectories.device, requires_grad=True)

        if self.reduction == 'mean':
            return nll_loss_unreduced.sum() / valid_agents_count
        elif self.reduction == 'sum':
            return nll_loss_unreduced.sum()
        else: # 'none'
            return nll_loss_unreduced
        
class FSDLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        # Instantiate individual loss components
        self.focal_loss_cls = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean') # For object detection classification and semantic segmentation
        self.focal_loss_det_cls = FocalLoss(gamma=2.0, alpha=0.5, reduction='mean') # A separate one for object classification if needed differently
        self.giou_loss = GIoULoss(reduction='mean') # For 3D object detection regression
        self.dice_loss = DiceLoss(reduction='mean') # For semantic segmentation and occupancy
        self.bce_loss = BCELoss(reduction='mean') # For occupancy (alternative to Dice)
        self.l1_loss = L1Loss(reduction='mean') # For depth estimation, planning trajectory
        self.l2_loss = L2Loss(reduction='mean') # For planning trajectory, potentially depth
        self.nll_loss_trajectory = NLLLossTrajectory(reduction='mean', sigma=0.5, use_best_mode_nll=True) # For multi-modal prediction

        # Define default weights. These would be tuned during hyperparameter optimization.
        self.weights = {
            'obj_cls': 1.0,
            'obj_reg': 2.0,
            'sem_seg': 1.0,
            'depth': 1.0,
            'occupancy': 0.5,
            'prediction_nll': 1.0,
            'planning_l1': 1.0, # For chosen trajectory
            'planning_action_cls': 0.1, # For high-level action
        }
        if weights is not None:
            self.weights.update(weights)

    def forward(self, model_outputs: dict, ground_truths: dict) -> dict:
        total_loss = torch.tensor(0.0, device=model_outputs['perception_outputs']['fused_bev_features'].device, requires_grad=True)
        loss_breakdown = {}

        # --- Perception Losses ---
        # Object Detection Classification (logits)
        obj_cls_loss = self.focal_loss_det_cls(
            model_outputs['perception_outputs']['object_detections']['logits'],
            ground_truths['obj_cls_targets']
        )
        total_loss = total_loss + self.weights['obj_cls'] * obj_cls_loss
        loss_breakdown['obj_cls_loss'] = obj_cls_loss.item()

        # Object Detection Regression (3D Bounding Boxes) - using GIoU
        obj_reg_loss = self.giou_loss(
            model_outputs['perception_outputs']['object_detections']['reg_outputs'],
            ground_truths['obj_reg_targets'],
            ground_truths['obj_mask']
        )
        total_loss = total_loss + self.weights['obj_reg'] * obj_reg_loss
        loss_breakdown['obj_reg_loss'] = obj_reg_loss.item()

        # Semantic Segmentation
        sem_seg_loss = self.focal_loss_cls(
            model_outputs['perception_outputs']['semantic_segmentation'],
            ground_truths['sem_seg_targets']
        )
        total_loss = total_loss + self.weights['sem_seg'] * sem_seg_loss
        loss_breakdown['sem_seg_loss'] = sem_seg_loss.item()

        # Depth Estimation (L1 Loss)
        depth_loss = self.l1_loss(
            model_outputs['perception_outputs']['depth_map'],
            ground_truths['depth_targets']
        )
        total_loss = total_loss + self.weights['depth'] * depth_loss
        loss_breakdown['depth_loss'] = depth_loss.item()

        # Occupancy Prediction (BCE Loss on queried points)
        # Note: OccupancyNetwork outputs (B, N_query, 1) sigmoid probability, GT should be (B, N_query, 1)
        # For BCEWithLogitsLoss, input should be logits. Here, we assume the OccupancyNetwork's final Sigmoid is removed for training.
        # If OccupancyNetwork directly outputs probabilities (with Sigmoid), we need to use `F.binary_cross_entropy` instead of `F.binary_cross_entropy_with_logits`.
        # For this example, let's assume `model_outputs['perception_outputs']['occupancy_probabilities']` is actually logits before sigmoid.
        # Or, if it's probability, we can use `self.bce_loss(torch.logit(model_outputs['perception_outputs']['occupancy_probabilities']), ground_truths['occupancy_targets'])`
        occupancy_loss = self.bce_loss(
            model_outputs['perception_outputs']['occupancy_probabilities'].squeeze(-1), # Assuming N_query points
            ground_truths['occupancy_targets'].squeeze(-1)
        )
        total_loss = total_loss + self.weights['occupancy'] * occupancy_loss
        loss_breakdown['occupancy_loss'] = occupancy_loss.item()

        # --- Prediction Losses ---
        # Multi-modal Trajectory Prediction (NLL Loss)
        prediction_nll_loss = self.nll_loss_trajectory(
            model_outputs['prediction_outputs']['predicted_trajectories'],
            model_outputs['prediction_outputs']['mode_probabilities'],
            ground_truths['gt_agent_trajectories'],
            ground_truths['gt_agent_mask']
        )
        total_loss = total_loss + self.weights['prediction_nll'] * prediction_nll_loss
        loss_breakdown['prediction_nll_loss'] = prediction_nll_loss.item()

        # --- Planning Losses ---
        # Planning: High-Level Action Classification (Cross-Entropy)
        # Assuming planning_outputs['high_level_action_logits'] are logits for N_actions
        planning_action_cls_loss = F.cross_entropy(
            model_outputs['planning_outputs']['high_level_action_logits'],
            ground_truths['high_level_action_targets']
        )
        total_loss = total_loss + self.weights['planning_action_cls'] * planning_action_cls_loss
        loss_breakdown['planning_action_cls_loss'] = planning_action_cls_loss.item()

        # Planning: Selected Trajectory Regression (L1 Loss)
        planning_l1_loss = self.l1_loss(
            model_outputs['planning_outputs']['selected_trajectory'],
            ground_truths['planning_trajectory_targets']
        )
        total_loss = total_loss + self.weights['planning_l1'] * planning_l1_loss
        loss_breakdown['planning_l1_loss'] = planning_l1_loss.item()

        # Additional planning losses could include safety violation penalties, comfort metrics.
        # For this example, we'll stick to supervised imitation learning for trajectory and action.

        loss_breakdown['total_loss'] = total_loss.item()

        return {'total_loss': total_loss, 'loss_breakdown': loss_breakdown}
