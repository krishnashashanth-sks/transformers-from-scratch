import torch
from torch.utils.data import Dataset

class DummyFSDDataset(Dataset):
    def __init__(self, num_samples, params):
        self.num_samples = num_samples
        self.params = params

        # Common parameters
        self.batch_size = params['batch_size'] # Note: DataLoader will use its own batch_size
        self.num_frames_to_fuse = params['num_frames_to_fuse']
        self.num_cameras = params['num_cameras']
        self.cam_img_in_channels = params['cam_img_in_channels']
        self.cam_img_height = params['cam_img_height']
        self.cam_img_width = params['cam_img_width']
        self.lidar_voxel_in_channels = params['lidar_voxel_in_channels']
        self.lidar_voxel_z_dim = params['lidar_voxel_z_dim']
        self.lidar_voxel_xy_dim = params['lidar_voxel_xy_dim']
        self.radar_in_channels = params['radar_in_channels']
        self.radar_raw_h = params['radar_raw_h']
        self.radar_raw_w = params['radar_raw_w']
        self.occupancy_query_point_dim = params['occupancy_query_point_dim']
        self.num_query_points = params['num_query_points']
        self.max_num_agents = params['max_num_agents']
        self.agent_input_features_dim = params['agent_input_features_dim']
        self.ego_state_dim = params['ego_state_dim']
        self.bev_h = params['bev_h']
        self.bev_w = params['bev_w']

        # GT-specific params
        self.det_num_classes = params['det_num_classes']
        self.seg_num_semantic_classes = params['seg_num_semantic_classes']
        self.num_future_steps_pred = params['num_future_steps_pred']
        self.trajectory_point_dim_pred = params['trajectory_point_dim_pred']
        self.num_high_level_actions = params['num_high_level_actions']
        self.num_future_steps_plan = params['num_future_steps_plan']
        self.trajectory_point_dim_plan = params['trajectory_point_dim_plan']
        self.occupancy_output_dim = params['occupancy_output_dim']


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy inputs for FSDSystem
        cam_input_sequence = torch.randn(
            self.num_frames_to_fuse, self.num_cameras,
            self.cam_img_in_channels, self.cam_img_height, self.cam_img_width
        )
        lidar_input_sequence = torch.randn(
            self.num_frames_to_fuse, self.lidar_voxel_in_channels,
            self.lidar_voxel_z_dim, self.lidar_voxel_xy_dim, self.lidar_voxel_xy_dim
        )
        radar_input_sequence = torch.randn(
            self.num_frames_to_fuse, self.radar_in_channels,
            self.radar_raw_h, self.radar_raw_w
        )
        occupancy_query_points = (torch.rand(self.num_query_points, self.occupancy_query_point_dim) * 2 - 1)

        detected_agents_states_seq = (torch.rand(
            self.num_frames_to_fuse, self.max_num_agents, self.agent_input_features_dim
        ) * 2 - 1)
        detected_agents_states_seq[:, :, 0] = detected_agents_states_seq[:, :, 0] * 2 - 1 # x_norm
        detected_agents_states_seq[:, :, 1] = detected_agents_states_seq[:, :, 1] * 2 - 1 # y_norm

        ego_vehicle_state = torch.randn(self.ego_state_dim)

        model_inputs = {
            'cam_input_sequence': cam_input_sequence,
            'lidar_input_sequence': lidar_input_sequence,
            'radar_input_sequence': radar_input_sequence,
            'occupancy_query_points': occupancy_query_points,
            'detected_agents_states_seq': detected_agents_states_seq,
            'ego_vehicle_state': ego_vehicle_state
        }

        # Dummy ground truths for FSDLoss
        ground_truths = {
            'obj_cls_targets': torch.randint(0, self.det_num_classes, (self.bev_h, self.bev_w)),
            'obj_reg_targets': torch.randn(6, self.bev_h, self.bev_w),
            'obj_mask': torch.randint(0, 2, (1, self.bev_h, self.bev_w), dtype=torch.bool),
            'sem_seg_targets': torch.randint(0, self.seg_num_semantic_classes, (self.bev_h, self.bev_w)),
            'depth_targets': torch.randn(1, self.bev_h, self.bev_w),
            'occupancy_targets': torch.randint(0, 2, (self.num_query_points, self.occupancy_output_dim)).float(),
            'gt_agent_trajectories': torch.randn(self.max_num_agents, self.num_future_steps_pred, self.trajectory_point_dim_pred),
            'gt_agent_mask': (torch.rand(self.max_num_agents) > 0.1).float(),
            'high_level_action_targets': torch.randint(0, self.num_high_level_actions, (1,)).squeeze(0), # Ensure scalar if needed by F.cross_entropy
            'planning_trajectory_targets': torch.randn(self.num_future_steps_plan, self.trajectory_point_dim_plan)
        }

        return model_inputs, ground_truths