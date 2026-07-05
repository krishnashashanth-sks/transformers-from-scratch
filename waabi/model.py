import torch.nn as nn
from layers import PerceptionNet,PredictionNet,PlanningNet
import torch
import torch.nn.functional as F

class EndToEndModel(nn.Module):
    def __init__(self, perception_params, prediction_params, planning_params):
        super(EndToEndModel, self).__init__()
        self.perception_net = PerceptionNet(**perception_params)
        self.prediction_net = PredictionNet(**prediction_params)
        self.planning_net = PlanningNet(**planning_params)

    def forward(self, camera_input, lidar_input, radar_input, ego_vehicle_state_features):
        batch_size = camera_input.shape[0] # Define batch_size here

        # 1. Perception Module
        perception_outputs = self.perception_net(camera_input, lidar_input, radar_input)

        # Extract relevant perception outputs for prediction inputs
        # NOTE: The ContextEncoder expects one-hot encoded semantic map with `bev_channels` dimensions.
        # The PerceptionNet output is `semantic_map` which is (batch_size, output_bev_channels, BEV_H, BEV_W).
        # To make it compatible with the ContextEncoder, we'll convert it to one-hot for prediction.
        # This is a conceptual bridge for E2E. In a real system, the PerceptionNet's semantic head
        # output structure and the ContextEncoder's input expectation would be designed for direct compatibility.

        # First, ensure the semantic_map from perception is in class ID format (H, W) before one-hot encoding
        # PerceptionNet's semantic_map is (batch_size, N_classes, BEV_H, BEV_W). We need to convert it
        # back to class indices (batch_size, BEV_H, BEV_W) to then one-hot encode it for ContextEncoder
        # The ContextEncoder expects 5 channels, so we max over the N_classes to get a single class ID per pixel.
        _, class_indices = torch.max(perception_outputs['semantic_map'], dim=1) # (batch_size, BEV_H, BEV_W)

        # Now one-hot encode it for the ContextEncoder, which expects `bev_channels` (5 in our case)
        # The `num_classes` here should be `self.prediction_net.context_encoder.conv_layers[0].in_channels`
        # which is 5. So we one-hot encode to 5 classes.
        # Ensure the `num_classes` in F.one_hot matches the `bev_channels` of ContextEncoder.
        num_classes_for_one_hot = self.prediction_net.context_encoder.conv_layers[0].in_channels
        bev_semantic_map_for_prediction = F.one_hot(class_indices, num_classes=num_classes_for_one_hot).permute(0, 3, 1, 2).float()

        # Simplified conversion from 3d_boxes to agent_current_states and historical_trajectories
        num_agents_pred = 3 # Fixed based on CustomDataset agent templates
        agent_state_dim = 7 # x, y, z, yaw, vx, vy, vz
        historical_steps = 10

        perceived_agent_current_states = torch.zeros(batch_size, num_agents_pred, agent_state_dim, device=camera_input.device)
        historical_agent_trajectories = torch.zeros(batch_size, num_agents_pred, historical_steps, agent_state_dim, device=camera_input.device)

        # Populate from perception_outputs['3d_boxes'] (which contains (x,y,z,yaw,l,w,h,vx,vy,cls1,cls2,cls3))
        # The first 7 relevant for agent_state_dim. Assume velocity is (vx,vy,0) for vz.
        for i in range(min(num_agents_pred, perception_outputs['3d_boxes'].shape[1])):
            # Assuming 3d_boxes: x,y,z,yaw,l,w,h,vx,vy,cls1,cls2,cls3
            # Map to agent_state_dim: x,y,z,yaw,vx,vy,vz(0)
            perceived_agent_current_states[:, i, 0:4] = perception_outputs['3d_boxes'][:, i, 0:4] # x,y,z,yaw
            perceived_agent_current_states[:, i, 4:6] = perception_outputs['3d_boxes'][:, i, 7:9] # vx,vy
            # For historical_agent_trajectories, we'll just use current state for all steps for this conceptual run
            historical_agent_trajectories[:, i, :, :] = perceived_agent_current_states[:, i, :].unsqueeze(1).repeat(1, historical_steps, 1)

        # 2. Prediction Module
        prediction_outputs = self.prediction_net(
            perceived_agent_current_states,
            historical_agent_trajectories,
            bev_semantic_map_for_prediction, # Use the one-hot encoded version
            ego_vehicle_state_features # This comes directly from input
        )

        # 3. Planning Module
        # The PlanningNet's bev_processor expects a (batch_size, bev_channels, bev_height, bev_width)
        # We will reuse the one-hot encoded map from perception here as well.
        planning_outputs = self.planning_net(
            bev_semantic_map_for_prediction, # Reusing BEV map
            prediction_outputs['predicted_trajectories'],
            prediction_outputs['trajectory_confidences'],
            prediction_outputs['predicted_intentions'],
            ego_vehicle_state_features # Reusing ego state
        )

        return {
            'perception_outputs': perception_outputs,
            'prediction_outputs': prediction_outputs,
            'planning_outputs': planning_outputs
        }

