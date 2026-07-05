from torch.utils.data import Dataset
from models import EgoVehicleState,DynamicAgent,RoadMap,Lane,TrafficLight,CameraSensor,LiDARSensor,RadarSensor,GroundTruthGenerator
import numpy as np
import torch
import random
import torch.nn.functional as F

# --- CustomDataset update for E2E task_type and prediction dummy historical data ---
class CustomDataset(Dataset):
    def __init__(self, num_samples, task_type='perception', transform=None):
        super(CustomDataset, self).__init__()
        self.num_samples = num_samples
        self.task_type = task_type # 'perception', 'prediction', 'planning', 'e2e'
        self.transform = transform

        # Initialize simulation components - for simplicity, these are global for now.
        # In a real system, scenarios might be pre-generated or more complex.
        self.ego_state_template = EgoVehicleState(
            x=0.0, y=0.0, yaw=0.0, vx=10.0,
            steering_angle=0.0, throttle=0.5, brake=0.0 # Added default controls for planning GT
        )
        self.agent_templates = [
            DynamicAgent(agent_id='car_0', agent_type='vehicle', x=20.0, y=0.5, z=0.0, yaw=0.0, vx=9.0, vy=0.0, vz=0.0, intention='straight'),
            DynamicAgent(agent_id='car_1', agent_type='vehicle', x=15.0, y=-3.0, z=0.0, yaw=np.pi/4, vx=8.0, vy=0.0, vz=0.0, intention='lane_change_right'),
            DynamicAgent(agent_id='ped_0', agent_type='pedestrian', x=10.0, y=5.0, z=0.0, yaw=np.pi/2, vx=-1.0, vy=0.0, vz=0.0, length=0.5, width=0.5, intention='crossing')
        ]
        self.roadmap = RoadMap()
        self.roadmap.add_lane(Lane(lane_id='lane_0', waypoints=[[0, 0], [50, 0]], speed_limit=16.67))
        self.roadmap.add_traffic_light(TrafficLight(light_id='tl_0', x=40.0, y=0.5, z=5.0, orientation_yaw=0.0, state='green'))

        self.cam_front = CameraSensor(sensor_id='cam_f', mount_x=1.5, mount_y=0.0, mount_z=1.5, mount_yaw=0.0, fov_h=np.deg2rad(90), fov_v=np.deg2rad(60), image_width=640, image_height=480)
        self.lidar_top = LiDARSensor(sensor_id='lidar_t', mount_x=0.0, mount_y=0.0, mount_z=2.0, mount_yaw=0.0, max_range=100.0, num_points=1000)
        self.radar_front = RadarSensor(sensor_id='radar_f', mount_x=2.0, mount_y=0.0, mount_z=0.5, mount_yaw=0.0, max_range=150.0, fov_h=np.deg2rad(60))

        self.gt_generator = GroundTruthGenerator()

        self.context_encoder_bev_channels = 5 # Used for one-hot encoding semantic map
        self.bev_target_map_resolution = (10, 10)

    def __len__(self):
        return self.num_samples

    def _generate_random_scenario(self):
        # Randomize ego state slightly
        ego_state = EgoVehicleState(
            x=np.random.uniform(-10, 10),
            y=np.random.uniform(-5, 5),
            yaw=np.random.uniform(-np.pi/4, np.pi/4),
            vx=np.random.uniform(5, 15),
            vy=np.random.uniform(-1, 1),
            steering_angle=np.random.uniform(-0.5, 0.5),
            throttle=np.random.uniform(0.1, 0.8),
            brake=np.random.uniform(0.0, 0.2)
        )

        # Randomize agent states slightly
        agents = []
        for i, agent_template in enumerate(self.agent_templates):
            agent = DynamicAgent(
                agent_id=f'agent_{i}',
                agent_type=agent_template.agent_type,
                x=ego_state.x + np.random.uniform(10, 50),
                y=ego_state.y + np.random.uniform(-10, 10),
                z=0.0,
                yaw=np.random.uniform(-np.pi, np.pi),
                vx=np.random.uniform(agent_template.vx - 2, agent_template.vx + 2),
                vy=np.random.uniform(-1, 1),
                vz=0.0,
                length=agent_template.length, width=agent_template.width, height=agent_template.height,
                intention=random.choice(['straight', 'turn_left', 'turn_right', 'stop'])
            )
            agents.append(agent)

        # Randomize traffic light state
        for light_id in self.roadmap.traffic_lights:
            self.roadmap.traffic_lights[light_id].state = random.choice(['red', 'yellow', 'green'])

        return ego_state, agents, self.roadmap

    def __getitem__(self, idx):
        # Generate a random scenario for this item
        ego_state, agents, roadmap = self._generate_random_scenario()

        # Simulate sensor data
        camera_image = self.cam_front.simulate(ego_state, agents, roadmap)
        lidar_points = self.lidar_top.simulate(ego_state, agents, roadmap)
        radar_detections = self.radar_front.simulate(ego_state, agents)

        # Generate ground truth for all modules
        perception_gt_raw = self.gt_generator.generate_perception_gt(ego_state, agents, roadmap)
        prediction_gt_raw = self.gt_generator.generate_prediction_gt(ego_state, agents)
        planning_gt_raw = self.gt_generator.generate_planning_gt(ego_state, roadmap)

        # --- Common Data Preparation (used by all task types) ---
        # Camera input
        camera_input = torch.from_numpy(camera_image).permute(2, 0, 1).float() / 255.0 # HWC to CHW, normalize

        # LiDAR input
        lidar_input = torch.from_numpy(lidar_points).float()

        # Radar input (pad/truncate to fixed size)
        max_radar_detections = 50 # Example max
        radar_tensor = torch.zeros(max_radar_detections, 3, dtype=torch.float)
        for i, det in enumerate(radar_detections):
            if i < max_radar_detections:
                radar_tensor[i, 0] = det['range']
                radar_tensor[i, 1] = det['radial_velocity']
                radar_tensor[i, 2] = det['azimuth_angle']
        radar_input = radar_tensor

        # Ego-vehicle state features for Prediction/Planning
        ego_vehicle_state_features = torch.tensor([
            ego_state.x, ego_state.y, ego_state.z, ego_state.yaw, ego_state.vx, ego_state.vy
        ], dtype=torch.float)

        # Semantic Map input for ContextEncoder (one-hot encoded BEV map)
        bev_semantic_map_raw = torch.from_numpy(perception_gt_raw['semantic_map']).long()
        bev_semantic_map_resized = F.interpolate(
            bev_semantic_map_raw.unsqueeze(0).unsqueeze(0).float(), # Add batch and channel dimensions (1, 1, H, W)
            size=self.bev_target_map_resolution,
            mode='nearest'
        ).squeeze(0).squeeze(0).long() # Remove batch and channel dimensions, cast back to long
        bev_semantic_map_input_one_hot = F.one_hot(bev_semantic_map_resized, num_classes=self.context_encoder_bev_channels).permute(2, 0, 1).float()
        # Final shape: (C, H, W) e.g., (5, 10, 10) of float

        # --- Ground Truth Conversions ---
        # Perception GT
        max_boxes = 10
        num_bbox_reg_attrs = 9 # x,y,z,yaw,l,w,h,vx,vy
        num_bbox_classes = 3 # vehicle, pedestrian, cyclist
        gt_3d_boxes = torch.zeros(max_boxes, num_bbox_reg_attrs + num_bbox_classes, dtype=torch.float)
        for i, box in enumerate(perception_gt_raw['3d_boxes']):
            if i < max_boxes:
                gt_3d_boxes[i, 0:num_bbox_reg_attrs] = torch.tensor([box['x'], box['y'], box['z'], box['yaw'], box['length'], box['width'], box['height'], box['vx'], box['vy']], dtype=torch.float)
                if box['agent_type'] == 'vehicle': gt_3d_boxes[i, num_bbox_reg_attrs] = 1.0
                elif box['agent_type'] == 'pedestrian': gt_3d_boxes[i, num_bbox_reg_attrs+1] = 1.0

        gt_semantic_map = bev_semantic_map_resized # Use the resized semantic map for GT

        max_lanes = 3
        num_lane_points = 50
        gt_lane_boundaries = torch.zeros(max_lanes, num_lane_points, 2, dtype=torch.float)
        # This part requires more sophisticated conversion for lane boundaries

        # Prediction GT
        max_agents = len(self.agent_templates) # Assuming fixed max agents
        prediction_horizon = self.gt_generator.prediction_steps
        pose_dim = self.gt_generator.pose_dim # Should be 3 (x, y, yaw)
        num_intention_classes = 5 # Example: straight, left, right, stop, yield

        gt_future_trajectories = torch.zeros(max_agents, prediction_horizon, pose_dim, dtype=torch.float)
        gt_agent_intentions = torch.zeros(max_agents, dtype=torch.long)

        for i, agent_traj_info in enumerate(prediction_gt_raw['future_trajectories']):
            if i < max_agents:
                traj = torch.tensor(agent_traj_info['trajectory'], dtype=torch.float)
                gt_future_trajectories[i, :min(prediction_horizon, traj.shape[0]), :] = traj[:min(prediction_horizon, traj.shape[0]), :]

        for i, agent_intent_info in enumerate(prediction_gt_raw['agent_intentions']):
            if i < max_agents:
                intent_map = {'straight': 0, 'turn_left': 1, 'turn_right': 2, 'stop': 3, 'crossing': 4, 'lane_change_right': 0} # Map intentions to int
                gt_agent_intentions[i] = intent_map.get(agent_intent_info['intention'], 0) # Default to 'straight'

        # Planning GT
        target_trajectory_len = 30 # Example target length
        control_dim = 3 # steering, throttle, brake

        gt_optimal_trajectory = torch.zeros(target_trajectory_len, pose_dim, dtype=torch.float)
        gt_control_commands = torch.zeros(control_dim, dtype=torch.float)

        if len(planning_gt_raw['optimal_trajectory']) > 0:
            traj_len = len(planning_gt_raw['optimal_trajectory'])
            gt_optimal_trajectory[:min(target_trajectory_len, traj_len), :] = torch.tensor(planning_gt_raw['optimal_trajectory'][:min(target_trajectory_len, traj_len)], dtype=torch.float)

        gt_control_commands[0] = ego_state.steering_angle
        gt_control_commands[1] = ego_state.throttle
        gt_control_commands[2] = ego_state.brake

        # --- Task-specific Returns ---
        if self.task_type == 'perception':
            data = {
                'camera_input': camera_input,
                'lidar_input': lidar_input,
                'radar_input': radar_input,
                'gt_3d_boxes': gt_3d_boxes,
                'gt_semantic_map': gt_semantic_map,
                'gt_lane_boundaries': gt_lane_boundaries
            }
        elif self.task_type == 'prediction':
            # For prediction, bev_semantic_map input is one-hot encoded
            perceived_agent_current_states_pred = gt_3d_boxes[:, :, :7] # Simplified: Use GT 3d_boxes for current state (x,y,z,yaw,vx,vy,vz(0))
            # Historical trajectories: repeat current state for now
            historical_agent_trajectories_pred = perceived_agent_current_states_pred.unsqueeze(1).repeat(1, 10, 1)

            data = {
                'perceived_agent_current_states': perceived_agent_current_states_pred,
                'historical_agent_trajectories': historical_agent_trajectories_pred,
                'bev_semantic_map': bev_semantic_map_input_one_hot,
                'ego_vehicle_state_features': ego_vehicle_state_features,
                'gt_future_trajectories': gt_future_trajectories,
                'gt_agent_intentions': gt_agent_intentions
            }
        elif self.task_type == 'planning':
            # For planning, bev_semantic_map input is one-hot encoded, and prediction outputs are assumed.
            # For simplicity, we use random tensors as 'predicted' inputs to planning for the DataLoader.
            num_output_trajectories = 3
            predicted_trajectories_input = torch.rand(max_agents, num_output_trajectories, prediction_horizon, pose_dim, dtype=torch.float)
            trajectory_confidences_input = torch.rand(max_agents, num_output_trajectories, dtype=torch.float)
            predicted_intentions_input = torch.rand(max_agents, num_intention_classes, dtype=torch.float)

            data = {
                'bev_semantic_map': bev_semantic_map_input_one_hot,
                'predicted_trajectories': predicted_trajectories_input,
                'trajectory_confidences': trajectory_confidences_input,
                'predicted_intentions': predicted_intentions_input,
                'ego_vehicle_state': ego_vehicle_state_features,
                'gt_optimal_trajectory': gt_optimal_trajectory,
                'gt_control_commands': gt_control_commands
            }
        elif self.task_type == 'e2e':
            # For E2E, return all raw inputs and all ground truths for all modules
            data = {
                'camera_input': camera_input,
                'lidar_input': lidar_input,
                'radar_input': radar_input,
                'ego_vehicle_state_features': ego_vehicle_state_features, # Input to Prediction/Planning
                'gt_3d_boxes': gt_3d_boxes, # Perception GT
                'gt_semantic_map': gt_semantic_map, # Perception GT
                'gt_lane_boundaries': gt_lane_boundaries, # Perception GT
                'gt_future_trajectories': gt_future_trajectories, # Prediction GT
                'gt_agent_intentions': gt_agent_intentions, # Prediction GT
                'gt_optimal_trajectory': gt_optimal_trajectory, # Planning GT
                'gt_control_commands': gt_control_commands # Planning GT
            }
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        if self.transform:
            data = self.transform(data)

        return data


