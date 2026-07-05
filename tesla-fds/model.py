import torch.nn as nn
from layers import PerceptionModule,PredictionModule,PlanningModule
import torch

class FSDSystem(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Common parameters for modules
        self.num_frames_to_fuse = kwargs.get('num_frames_to_fuse')
        self.fusion_embed_dim = kwargs.get('fusion_embed_dim')
        self.bev_h = kwargs.get('bev_h')
        self.bev_w = kwargs.get('bev_w')
        self.det_num_classes = kwargs.get('det_num_classes')
        self.det_num_regression_params = kwargs.get('det_num_regression_params')
        self.seg_num_semantic_classes = kwargs.get('seg_num_semantic_classes')
        self.occupancy_query_point_dim = kwargs.get('occupancy_query_point_dim')
        self.num_modes = kwargs.get('num_modes')
        self.num_future_steps_pred = kwargs.get('num_future_steps_pred')
        self.trajectory_point_dim_pred = kwargs.get('trajectory_point_dim_pred')
        self.num_high_level_actions = kwargs.get('num_high_level_actions')
        self.num_candidate_trajectories = kwargs.get('num_candidate_trajectories')
        self.num_future_steps_plan = kwargs.get('num_future_steps_plan')
        self.trajectory_point_dim_plan = kwargs.get('trajectory_point_dim_plan')


        # 1. Instantiate Perception Module
        self.perception_module = PerceptionModule(
            num_cameras=kwargs.get('num_cameras'),
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            num_frames_to_fuse=self.num_frames_to_fuse,
            cam_img_in_channels=kwargs.get('cam_img_in_channels'),
            cam_backbone_out_channels=kwargs.get('cam_backbone_out_channels'),
            cam_fpn_out_channels=kwargs.get('cam_fpn_out_channels'),
            cam_bev_channels=kwargs.get('cam_bev_channels'),
            lidar_voxel_in_channels=kwargs.get('lidar_voxel_in_channels'),
            lidar_bev_channels=kwargs.get('lidar_bev_channels'),
            radar_in_channels=kwargs.get('radar_in_channels'),
            radar_bev_channels=kwargs.get('radar_bev_channels'),
            fusion_embed_dim=self.fusion_embed_dim,
            fusion_num_heads=kwargs.get('fusion_num_heads'),
            fusion_num_layers=kwargs.get('fusion_num_layers'),
            fusion_dropout=kwargs.get('fusion_dropout'),
            occupancy_query_point_dim=self.occupancy_query_point_dim,
            occupancy_hidden_dim=kwargs.get('occupancy_hidden_dim'),
            occupancy_output_dim=kwargs.get('occupancy_output_dim'),
            det_num_classes=self.det_num_classes,
            det_num_regression_params=self.det_num_regression_params,
            det_hidden_dim=kwargs.get('det_hidden_dim'),
            seg_num_semantic_classes=self.seg_num_semantic_classes,
            seg_hidden_dim=kwargs.get('seg_hidden_dim'),
            depth_hidden_dim=kwargs.get('depth_hidden_dim'),
        )

        # 2. Instantiate Prediction Module
        self.prediction_module = PredictionModule(
            fused_bev_channels=self.fusion_embed_dim,
            agent_input_features_dim=kwargs.get('agent_input_features_dim'),
            agent_bev_patch_size=kwargs.get('agent_bev_patch_size'),
            agent_feature_extractor_output_dim=kwargs.get('agent_feature_extractor_output_dim'),
            num_frames_to_fuse=self.num_frames_to_fuse,
            temporal_encoder_max_frames=kwargs.get('temporal_encoder_max_frames'),
            temporal_encoder_embed_dim=kwargs.get('temporal_encoder_embed_dim'),
            interaction_model_embed_dim=kwargs.get('interaction_model_embed_dim'),
            interaction_model_max_agents=kwargs.get('interaction_model_max_agents'),
            num_future_steps=self.num_future_steps_pred,
            num_modes=self.num_modes,
            trajectory_point_dim=self.trajectory_point_dim_pred,
            temporal_encoder_num_heads=kwargs.get('temporal_encoder_num_heads'),
            temporal_encoder_num_layers=kwargs.get('temporal_encoder_num_layers'),
            temporal_encoder_dropout=kwargs.get('temporal_encoder_dropout'),
            interaction_model_num_heads=kwargs.get('interaction_model_num_heads'),
            interaction_model_num_layers=kwargs.get('interaction_model_num_layers'),
            interaction_model_dropout=kwargs.get('interaction_model_dropout'),
            trajectory_decoder_hidden_dim=kwargs.get('trajectory_decoder_hidden_dim'),
        )

        # 3. Instantiate Planning Module
        # The planning_context_encoder_input_dim needs to be calculated dynamically after knowing all component outputs.
        # We will calculate a proxy here based on expected flattened output sizes.
        # Perception Outputs:
        # - fused_bev_features: B, fusion_embed_dim, bev_h, bev_w
        # - occupancy_probabilities: B, num_query_points, 1
        # - semantic_segmentation: B, seg_num_semantic_classes, bev_h, bev_w
        # Prediction Outputs:
        # - predicted_trajectories: B, N_agents, num_modes, num_future_steps_pred, trajectory_point_dim_pred
        # - mode_probabilities: B, N_agents, num_modes
        # Ego State: B, ego_state_dim

        _fused_bev_flat_size = self.fusion_embed_dim * self.bev_h * self.bev_w
        _occupancy_flat_size = kwargs.get('num_query_points') * kwargs.get('occupancy_output_dim') # Assumed occupancy_output_dim is 1
        _semantic_flat_size = self.seg_num_semantic_classes * self.bev_h * self.bev_w
        _predicted_traj_flat_size = kwargs.get('max_num_agents') * self.num_modes * self.num_future_steps_pred * self.trajectory_point_dim_pred
        _mode_prob_flat_size = kwargs.get('max_num_agents') * self.num_modes
        _ego_state_flat_size = kwargs.get('ego_state_dim')

        planning_context_encoder_input_dim = (
            _fused_bev_flat_size + _occupancy_flat_size + _semantic_flat_size +
            _predicted_traj_flat_size + _mode_prob_flat_size + _ego_state_flat_size
        )

        self.planning_module = PlanningModule(
            context_encoder_input_dim=planning_context_encoder_input_dim,
            context_embedding_dim=kwargs.get('planning_context_embedding_dim'),
            num_high_level_actions=self.num_high_level_actions,
            num_candidate_trajectories=self.num_candidate_trajectories,
            num_future_steps=self.num_future_steps_plan,
            trajectory_point_dim=self.trajectory_point_dim_plan,
            hidden_dim=kwargs.get('planning_hidden_dim'),
        )

    def forward(self,
                cam_input_sequence: torch.Tensor, # (B, T, num_cameras, C_img, H_img, W_img)
                lidar_input_sequence: torch.Tensor, # (B, T, C_voxel, Z_voxel, Y_voxel, X_voxel)
                radar_input_sequence: torch.Tensor, # (B, T, C_radar_raw, H_radar_raw, W_radar_raw)
                occupancy_query_points: torch.Tensor, # (B, N_query, query_point_dim) for current frame occupancy
                detected_agents_states_seq: torch.Tensor, # (B, T, N_agents, F_agent_input) for prediction history
                ego_vehicle_state: torch.Tensor # (B, ego_state_dim) for current frame
                ) -> dict:
        B, T_seq, _, _, _, _ = cam_input_sequence.shape # T_seq should be num_frames_to_fuse

        # --- PERCEPTION MODULE ----
        # Processes sequence of raw sensor inputs and returns current frame outputs (fused BEV, detections, etc.)
        perception_outputs = self.perception_module(
            cam_input_sequence,
            lidar_input_sequence,
            radar_input_sequence,
            occupancy_query_points
        )

        current_fused_bev_features = perception_outputs['fused_bev_features']
        current_semantic_segmentation = perception_outputs['semantic_segmentation']
        current_occupancy_probabilities = perception_outputs['occupancy_probabilities']
        # object_detections from perception_outputs are not directly used as input to PredictionModule,
        # instead `detected_agents_states_seq` is provided as an input to FSDSystem.

        # --- PREDICTION MODULE ---
        # Requires fused_bev_features_seq. Simplification: repeat current_fused_bev_features T_seq times.
        # In a real system, the PerceptionModule would output a sequence or a history buffer would be maintained.
        fused_bev_features_seq_for_prediction = current_fused_bev_features.unsqueeze(1).repeat(1, T_seq, 1, 1, 1)

        prediction_outputs = self.prediction_module(
            fused_bev_features_seq_for_prediction,
            detected_agents_states_seq
        )

        predicted_trajectories = prediction_outputs['predicted_trajectories']
        mode_probabilities = prediction_outputs['mode_probabilities']

        # --- PLANNING MODULE ---
        planning_outputs = self.planning_module(
            fused_bev_features=current_fused_bev_features,
            occupancy_probabilities=current_occupancy_probabilities,
            semantic_segmentation=current_semantic_segmentation,
            predicted_trajectories=predicted_trajectories,
            mode_probabilities=mode_probabilities,
            ego_vehicle_state=ego_vehicle_state
        )

        return {
            "perception_outputs": perception_outputs,
            "prediction_outputs": prediction_outputs,
            "planning_outputs": planning_outputs
        }