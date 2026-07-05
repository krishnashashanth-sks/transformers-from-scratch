import torch.nn as nn
import torch
import torch.nn.functional as F

class CameraEncoder(nn.Module):
  def __init__(self,in_channels=3,image_width=640,image_height=480,feature_dim=512):
    super(CameraEncoder,self).__init__()
    self.conv_layers=nn.Sequential(
        nn.Conv2d(in_channels,32,kernel_size=3,stride=2,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
        nn.ReLU(),
        nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
        nn.ReLU()
    )
    dummy_input=torch.zeros(1,in_channels,image_height,image_width)
    with torch.no_grad():
      self._num_features=self.conv_layers(dummy_input).numel()
    self.fc=nn.Linear(self._num_features,feature_dim)
  def forward(self,x):
    x=self.conv_layers(x)
    x=torch.flatten(x,1)
    return self.fc(x)
  
class LiDAREncoder(nn.Module):
    """Encodes LiDAR point cloud data into a feature vector."""
    def __init__(self, num_points=1000, features_per_point=4, feature_dim=512):
        super(LiDAREncoder, self).__init__()
        # For simplicity, treat each point's features as input to a linear layer
        # A more sophisticated model would use PointNet-like architectures.
        self.fc1 = nn.Linear(features_per_point, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x shape: (batch_size, num_points, features_per_point)
        # Apply linear layer to each point feature individually
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Aggregate features across points (e.g., max pooling)
        x = torch.max(x, dim=1).values # Output shape: (batch_size, feature_dim)
        return x

class RadarEncoder(nn.Module):
    """Encodes radar detection data into a feature vector."""
    def __init__(self, features_per_detection=3, feature_dim=512):
        super(RadarEncoder, self).__init__()
        # features_per_detection: (range, radial_velocity, azimuth_angle)
        self.fc1 = nn.Linear(features_per_detection, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x shape: (batch_size, num_detections, features_per_detection)
        # Apply linear layer to each detection's features individually
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Aggregate features across detections (e.g., max pooling)
        x = torch.max(x, dim=1).values # Output shape: (batch_size, feature_dim)
        return x

class PerceptionNet(nn.Module):
  def __init__(self,camera_in_channels=3,camera_image_width=640,camera_image_height=480,camera_feature_dim=512,
               lidar_num_points=1000,lidar_features_per_point=4,lidar_feature_dim=512,
               radar_features_per_detection=3,radar_feature_dim=512,
               output_bev_channels=64,num_classes_boxes=3,num_lane_points=50):
    super(PerceptionNet,self).__init__()
    self.camera_encoder=CameraEncoder(in_channels=camera_in_channels,
                                            image_width=camera_image_width,
                                            image_height=camera_image_height,
                                            feature_dim=camera_feature_dim)
    self.lidar_encoder = LiDAREncoder(num_points=lidar_num_points,
                                          features_per_point=lidar_features_per_point,
                                          feature_dim=lidar_feature_dim)
    self.radar_encoder = RadarEncoder(features_per_detection=radar_features_per_detection,
                                          feature_dim=radar_feature_dim)
    self.fusion_dim=camera_feature_dim+lidar_feature_dim+radar_feature_dim
    self.fusion_layer=nn.Sequential(
        nn.Linear(self.fusion_dim,1024),
        nn.ReLU(),
        nn.Linear(1024,512)
    )
    self.bev_projection=nn.Sequential(
        nn.Linear(512,output_bev_channels*10*10),
        nn.ReLU()
    )
    self.bev_h=10
    self.bev_W=10
    self.output_bev_channels=output_bev_channels
    self.bbox_head=nn.Sequential(
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Linear(256,10*(9+num_classes_boxes))
    )
    self.num_classes_boxes=num_classes_boxes
    self.semantic_head=nn.Conv2d(output_bev_channels,5,kernel_size=1)
    self.lane_head=nn.Sequential(
        nn.Flatten(),
        nn.Linear(output_bev_channels*self.bev_h*self.bev_W,256),
        nn.ReLU(),
        nn.Linear(256,3*num_lane_points*2)
    )
    self.num_lane_types=3
    self.num_lane_points=num_lane_points
  def forward(self,camera_input,lidar_input,radar_input):
    camera_features=self.camera_encoder(camera_input)
    lidar_features=self.lidar_encoder(lidar_input)
    radar_features=self.radar_encoder(radar_input)
    fused_features=torch.cat((camera_features,lidar_features,radar_features),dim=1)
    fused_features=self.fusion_layer(fused_features)
    bev_features=self.bev_projection(fused_features)
    bev_features=bev_features.view(-1,self.output_bev_channels,self.bev_h,self.bev_W)
    bbox_predictions=self.bbox_head(fused_features)
    bbox_predictions=bbox_predictions.view(fused_features.size(0),10,-1)
    semantic_map=self.semantic_head(bev_features)
    lane_predictions=self.lane_head(bev_features)
    lane_predictions=lane_predictions.view(fused_features.size(0),self.num_lane_types,self.num_lane_points,2)
    return {
            '3d_boxes': bbox_predictions,      # (batch_size, N_objects, Attr_per_object)
            'semantic_map': semantic_map,      # (batch_size, N_classes, BEV_H, BEV_W)
            'lane_boundaries': lane_predictions # (batch_size, N_lanes, N_points, 2)
    }
  
class AgentFeatureEncoder(nn.Module):
  def __init__(self,agent_state_dim=6,historical_steps=10,hidden_dim=128):
    super(AgentFeatureEncoder,self).__init__()
    self.historical_steps=historical_steps
    self.rnn=nn.GRU(agent_state_dim,hidden_dim,batch_first=True)
    self.fc_current_state=nn.Linear(agent_state_dim,hidden_dim)
    self.output_dim=hidden_dim*2
  def forward(self,current_state_features,historical_trajectory_features):
    batch_size,num_agents,_=current_state_features.shape
    current_features=F.relu(self.fc_current_state(current_state_features))
    historical_trajectory_features_reshaped=historical_trajectory_features.view(
        batch_size*num_agents,self.historical_steps,-1
    )
    _,historical_features=self.rnn(historical_trajectory_features_reshaped)
    historical_features=historical_features.view(batch_size,num_agents,-1)
    return torch.cat((current_features,historical_features),dim=-1)
  
class ContextEncoder(nn.Module):
    """Encodes environmental context (BEV semantic map) into a feature vector."""
    def __init__(self, bev_channels=5, bev_height=10, bev_width=10, feature_dim=256):
        super(ContextEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(bev_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Downsample
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Downsample
            nn.ReLU()
        )

        # Calculate flattened feature size
        dummy_input = torch.zeros(1, bev_channels, bev_height, bev_width)
        with torch.no_grad():
            self._num_features = self.conv_layers(dummy_input).numel()

        self.fc = nn.Linear(self._num_features, feature_dim)

    def forward(self, bev_semantic_map):
        # bev_semantic_map: (batch_size, bev_channels, bev_height, bev_width)
        x = self.conv_layers(bev_semantic_map)
        x = torch.flatten(x, 1)
        context_features = self.fc(x)
        return context_features # Output: (batch_size, feature_dim)

print("ContextEncoder class defined.")
class InteractionModule(nn.Module):
  def __init__(self,agent_feature_dim,context_feature_dim,hidden_dim=256,num_heads=4):
    super(InteractionModule,self).__init__()
    self.agent_feature_dim=agent_feature_dim
    self.context_feature_dim=context_feature_dim
    self.hidden_dim=hidden_dim
    self.agent_projection=nn.Linear(agent_feature_dim,hidden_dim)
    self.context_projection=nn.Linear(context_feature_dim,hidden_dim)
    self.attention=nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=num_heads,batch_first=True)
    self.ffn=nn.Sequential(
        nn.Linear(hidden_dim,hidden_dim*2),
        nn.ReLU(),
        nn.Linear(hidden_dim*2,hidden_dim)
    )
    self.norm1=nn.LayerNorm(hidden_dim)
    self.norm2=nn.LayerNorm(hidden_dim)
  def forward(self,agent_features,context_features):
    batch_size,num_agents,_=agent_features.shape
    agent_qkv=self.agent_projection(agent_features)
    context_kv=self.context_projection(context_features).unsqueeze(1).repeat(1,num_agents,1)
    attended_context,_=self.attention(query=agent_qkv,key=context_kv,value=context_kv)
    contextualized_agent_features,_=self.attention(query=agent_qkv,key=context_kv,value=context_kv)
    agent_features_interacted=self.norm1(agent_qkv+contextualized_agent_features)
    return self.norm2(agent_features_interacted+self.ffn(agent_features_interacted))
  
class TrajectoryDecoder(nn.Module):
  def __init__(self,input_feature_dim,num_output_trajectories=3,prediction_horizon=20,
               pose_dim=3,num_intention_classes=5):
    super(TrajectoryDecoder,self).__init__()
    self.num_output_trajectories=num_output_trajectories
    self.prediction_horizon=prediction_horizon
    self.pose_dim=pose_dim
    self.num_intention_classes=num_intention_classes
    self.trajectory_head=nn.Sequential(
        nn.Linear(input_feature_dim,256),
        nn.ReLU(),
        nn.Linear(256,num_output_trajectories*(prediction_horizon*pose_dim+1))
    )
    self.intention_head=nn.Sequential(
        nn.Linear(input_feature_dim,128),
        nn.ReLU(),
        nn.Linear(128,num_intention_classes)
    )
  def forward(self,interacted_agent_features):
    batch_size,num_agents,_=interacted_agent_features.shape
    trajectories_raw=self.trajectory_head(interacted_agent_features)
    trajectories_raw=trajectories_raw.view(
        batch_size,num_agents,self.num_output_trajectories,self.prediction_horizon*self.pose_dim+1
    )
    predicted_trajectories=trajectories_raw[...,:-1].contiguous().view(
        batch_size,num_agents,self.num_output_trajectories,self.prediction_horizon,self.pose_dim
    )
    trajectory_confidences=torch.softmax(trajectories_raw[...,-1],dim=-1)
    intention_logits=self.intention_head(interacted_agent_features)
    predicted_intentions=torch.softmax(intention_logits,dim=-1)
    return {
            'predicted_trajectories': predicted_trajectories,  # (batch_size, num_agents, num_output_trajectories, prediction_horizon, pose_dim)
            'trajectory_confidences': trajectory_confidences,  # (batch_size, num_agents, num_output_trajectories)
            'predicted_intentions': predicted_intentions       # (batch_size, num_agents, num_intention_classes)
        }
  
class PredictionNet(nn.Module):
  def __init__(self,
               agent_state_dim=7,historical_steps=10,agent_hidden_dim=128,
               bev_channels=5,bev_height=10,bev_width=10,
               context_feature_dim=256,
               interaction_hidden_dim=256,
               interaction_num_heads=4,
               num_output_trajectories=3,
               prediction_horizon=20,
               pose_dim=3,
               num_intention_classes=5,
               ego_feature_dim=6):
    super(PredictionNet,self).__init__()
    self.agent_encoder=AgentFeatureEncoder(
        agent_state_dim=agent_state_dim,
        historical_steps=historical_steps,
        hidden_dim=agent_hidden_dim
    )
    self.agent_encoder_feature_dim=self.agent_encoder.output_dim
    self.context_encoder=ContextEncoder(
        bev_channels=bev_channels,
        bev_height=bev_height,
        bev_width=bev_width,
        feature_dim=context_feature_dim
    )
    self.global_context_fusion_dim=ego_feature_dim+context_feature_dim
    self.global_context_projection=nn.Sequential(
        nn.Linear(self.global_context_fusion_dim,interaction_hidden_dim),
        nn.ReLU()
    )
    self.interaction_module=InteractionModule(
        agent_feature_dim=self.agent_encoder_feature_dim,
        context_feature_dim=interaction_hidden_dim,
        hidden_dim=interaction_hidden_dim,
        num_heads=interaction_num_heads
    )
    self.trajectory_decoder=TrajectoryDecoder(
        input_feature_dim=interaction_hidden_dim,
        num_output_trajectories=num_output_trajectories,
        prediction_horizon=prediction_horizon,
        pose_dim=pose_dim,
        num_intention_classes=num_intention_classes
    )
  def forward(self,perceived_agent_current_states,historical_agent_trajectories,
              bev_semantic_map,ego_vehicle_state_features):
    agent_features=self.agent_encoder(perceived_agent_current_states,historical_agent_trajectories)
    context_features=self.context_encoder(bev_semantic_map)
    global_context_raw=torch.cat(
        (ego_vehicle_state_features,context_features),
        dim=-1
    )
    global_context=self.global_context_projection(global_context_raw)
    interacted_agent_features=self.interaction_module(agent_features,global_context)
    return self.trajectory_decoder(interacted_agent_features)
  
class PlanningNet(nn.Module):
  def __init__(self,bev_channels,bev_height=10,bev_width=10, # Corrected bev_height from 20 to 10 for consistency with dummy inputs
               num_agents=5,num_output_trajectories=3,prediction_horizon=20,pose_dim=3,num_intention_classes=5,
               ego_state_dim=6,
               target_trajectory_len=30,control_dim=3,
               hidden_dim=256):
    super(PlanningNet,self).__init__()
    self.pose_dim=pose_dim
    self.target_trajectory_len=target_trajectory_len
    self.control_dim=control_dim
    self.bev_processor=nn.Sequential(
        nn.Conv2d(bev_channels,32,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*bev_height*bev_width,hidden_dim) # Corrected input dimension
    )
    agent_feature_input_dim=(
      num_output_trajectories*prediction_horizon*pose_dim+
      num_output_trajectories+
      num_intention_classes
    )
    self.agent_processor=nn.Sequential(
        nn.Linear(agent_feature_input_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim)
    )
    self.agent_aggregator=nn.Linear(num_agents*hidden_dim,hidden_dim)
    self.ego_processor=nn.Sequential(
        nn.Linear(ego_state_dim,hidden_dim//2),
        nn.ReLU(),
        nn.Linear(hidden_dim//2,hidden_dim//2)
    )
    self.fusion_input_dim=hidden_dim+hidden_dim+(hidden_dim//2)
    self.fusion_layer=nn.Sequential(
        nn.Linear(self.fusion_input_dim,hidden_dim*2),
        nn.ReLU(),
        nn.Linear(hidden_dim*2,hidden_dim)
    )
    self.trajectory_generation_head=nn.Sequential(
        nn.Linear(hidden_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,target_trajectory_len*pose_dim)
    )
    self.control_command_head=nn.Sequential(
        nn.Linear(hidden_dim,hidden_dim//2),
        nn.ReLU(),
        nn.Linear(hidden_dim//2,control_dim)
    )
  def forward(self,bev_semantic_map,predicted_trajectories,trajectory_confidences,predicted_intentions,ego_vehicle_state):
    batch_size=bev_semantic_map.shape[0]
    bev_features=self.bev_processor(bev_semantic_map)

    # Corrected trajectory flattening:
    predicted_trajectories_flat=predicted_trajectories.view(
        batch_size,predicted_trajectories.shape[1],-1
    )

    agent_raw_features=torch.cat([
        predicted_trajectories_flat,
        trajectory_confidences,
        predicted_intentions
    ],dim=-1)
    processed_agent_features=self.agent_processor(agent_raw_features)
    aggregated_agent_features=self.agent_aggregator(processed_agent_features.view(batch_size,-1))
    ego_features=self.ego_processor(ego_vehicle_state)
    fused_planning_context=self.fusion_layer(torch.cat([
      bev_features,
      aggregated_agent_features,
      ego_features
    ],dim=-1))
    optimal_trajectory=self.trajectory_generation_head(fused_planning_context)
    optimal_trajectory=optimal_trajectory.view(batch_size,self.target_trajectory_len,self.pose_dim)
    control_commands=self.control_command_head(fused_planning_context)
    return {
            'optimal_trajectory': optimal_trajectory,  # (batch_size, target_trajectory_len, pose_dim)
            'control_commands': control_commands       # (batch_size, control_dim)

            }