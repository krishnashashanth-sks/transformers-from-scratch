import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Start of Helper Modules (re-defined for completeness) -----------------

class SwinTransformerBackbone(nn.Module):
  def __init__(self,in_channels,out_channels=256):
    super().__init__()
    self.conv1=nn.Conv2d(in_channels,out_channels//4,kernel_size=3,stride=2,padding=1)
    self.conv2=nn.Conv2d(out_channels//4,out_channels//2,kernel_size=3,stride=2,padding=1)
    self.conv3=nn.Conv2d(out_channels//2,out_channels,kernel_size=3,stride=2,padding=1)
  def forward(self,x):
    f1=self.conv1(x)
    f2=self.conv2(f1)
    f3=self.conv3(f2)
    return [f1,f2,f3]

class FPN(nn.Module):
  def __init__(self,in_channels_list,out_channels):
    super().__init__()
    self.lat_convs=nn.ModuleList()
    self.out_convs=nn.ModuleList()
    for in_channels in in_channels_list:
      self.lat_convs.append(nn.Conv2d(in_channels,out_channels,kernel_size=1))
      self.out_convs.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
  def forward(self,features):
    out_features=[]
    # Iterate from deepest to shallowest feature map
    for i in range(len(features)-1,-1,-1):
      if i==len(features)-1:
        x=self.lat_convs[i](features[i])
      else:
        # Upsample previous output and add to current lateral connection
        x=self.lat_convs[i](features[i])+F.interpolate(out_features[-1],size=features[i].shape[2:],mode='nearest')
      out_features.append(self.out_convs[i](x))
    return out_features[::-1] # Reverse to get features from shallowest to deepest

class ViewTransformer(nn.Module):
  def __init__(self,input_dim,output_res_h,output_res_w,output_channels):
    super().__init__()
    self.output_res_h=output_res_h
    self.output_res_w=output_res_w
    self.output_channels=output_channels
    self.bev_projection=nn.Conv2d(input_dim,output_channels,kernel_size=1)
  def forward(self,img_features):
    bev_features=F.interpolate(img_features,size=(self.output_res_h,self.output_res_w),mode='bilinear',align_corners=False)
    return self.bev_projection(bev_features)

class CameraEncoder(nn.Module):
  def __init__(self,in_channels,backbone_out_channels,fpn_out_channels,bev_h,bev_w,bev_channels):
    super().__init__()
    self.backbone=SwinTransformerBackbone(in_channels,out_channels=backbone_out_channels)
    self.fpn=FPN([backbone_out_channels//4,backbone_out_channels//2,backbone_out_channels],fpn_out_channels)
    self.bev_h=bev_h
    self.bev_w=bev_w
    self.view_transformer=ViewTransformer(fpn_out_channels,bev_h,bev_w,bev_channels)
  def forward(self,x):
    batch_size,num_cameras,C,H,W=x.shape
    all_bev_features=[]
    for i in range(num_cameras):
      cam_features=self.backbone(x[:,i,:,:,:])
      fpn_features=self.fpn(cam_features)
      bev_features=self.view_transformer(fpn_features[0]) # Using the highest-resolution FPN feature
      all_bev_features.append(bev_features)
    return torch.cat(all_bev_features,dim=1) # Concatenate features from all cameras

class SparseConvNet(nn.Module):
  def __init__(self,in_channels,bev_channels):
    super().__init__()
    self.conv3d_1=nn.Conv3d(in_channels,32,kernel_size=3,padding=1)
    self.conv3d_2=nn.Conv3d(32,64,kernel_size=3,padding=1)
    self.conv2d_bev=nn.Conv2d(64*8,bev_channels,kernel_size=1) # Assuming Z_dim=8
  def forward(self,voxels):
    x=F.relu(self.conv3d_1(voxels))
    x=F.relu(self.conv3d_2(x))
    batch_size,C,Z,Y,X=x.shape
    bev_features=x.view(batch_size,C*Z,Y,X) # Flatten Z dimension into channels
    return self.conv2d_bev(bev_features)

class LiDAREncoder(nn.Module):
  def __init__(self,in_voxel_channels,bev_channels,bev_h,bev_w):
    super().__init__()
    self.sparse_conv_net=SparseConvNet(in_voxel_channels,bev_channels)
    self.bev_h=bev_h
    self.bev_w=bev_w
  def forward(self,voxel_input):
    bev_features=self.sparse_conv_net(voxel_input)
    if bev_features.shape[2]!=self.bev_h or bev_features.shape[3]!=self.bev_w:
      bev_features=F.interpolate(bev_features,size=(self.bev_h,self.bev_w),mode='bilinear',align_corners=False)
    return bev_features

class RadarEncoder(nn.Module):
    def __init__(self, in_channels, bev_channels, bev_h, bev_w):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, bev_channels, kernel_size=1)
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, radar_bev_input):
        x = F.relu(self.conv1(radar_bev_input))
        x = F.relu(self.conv2(x))
        x = self.conv_out(x)
        if x.shape[2] != self.bev_h or x.shape[3] != self.bev_w:
            x = F.interpolate(x, size=(self.bev_h, self.bev_w), mode='bilinear', align_corners=False)
        return x

class MultiModalTemporalFusion(nn.Module):
    def __init__(self,
                 cam_bev_input_channels,
                 lidar_bev_input_channels,
                 radar_bev_input_channels,
                 bev_h, bev_w,
                 fusion_embed_dim,
                 num_frames_to_fuse,
                 num_heads=8,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fusion_embed_dim = fusion_embed_dim
        self.num_frames_to_fuse = num_frames_to_fuse

        self.cam_proj = nn.Conv2d(cam_bev_input_channels, fusion_embed_dim, kernel_size=1)
        self.lidar_proj = nn.Conv2d(lidar_bev_input_channels, fusion_embed_dim, kernel_size=1)
        self.radar_proj = nn.Conv2d(radar_bev_input_channels, fusion_embed_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_embed_dim, nhead=num_heads,
                                                   dim_feedforward=fusion_embed_dim * 2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, fusion_embed_dim, bev_h, bev_w))
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, num_frames_to_fuse, fusion_embed_dim, 1, 1))
        self.output_proj = nn.Conv2d(fusion_embed_dim, fusion_embed_dim, kernel_size=1)

    def forward(self, cam_bev_features_seq, lidar_bev_features_seq, radar_bev_features_seq):
        B, T, C_cam_bev, H_bev, W_bev = cam_bev_features_seq.shape
        _, _, C_lidar_bev, _, _ = lidar_bev_features_seq.shape
        _, _, C_radar_bev, _, _ = radar_bev_features_seq.shape

        assert T == self.num_frames_to_fuse, f"Number of input frames ({T}) does not match num_frames_to_fuse ({self.num_frames_to_fuse})"

        cam_proj_features = self.cam_proj(cam_bev_features_seq.view(B * T, C_cam_bev, H_bev, W_bev)).view(B, T, self.fusion_embed_dim, H_bev, W_bev)
        lidar_proj_features = self.lidar_proj(lidar_bev_features_seq.view(B * T, C_lidar_bev, H_bev, W_bev)).view(B, T, self.fusion_embed_dim, H_bev, W_bev)
        radar_proj_features = self.radar_proj(radar_bev_features_seq.view(B * T, C_radar_bev, H_bev, W_bev)).view(B, T, self.fusion_embed_dim, H_bev, W_bev)

        fused_features_with_pe = (
            cam_proj_features +
            lidar_proj_features +
            radar_proj_features +
            self.spatial_pos_embedding.unsqueeze(1) +
            self.temporal_pos_embedding
        )

        seq_len = T * H_bev * W_bev
        transformer_input = fused_features_with_pe.permute(0, 1, 3, 4, 2).reshape(B, seq_len, self.fusion_embed_dim)
        transformer_output = self.transformer_encoder(transformer_input)

        output_fused_tokens = transformer_output.reshape(B, T, H_bev, W_bev, self.fusion_embed_dim)

        # Return features for the current (last) frame
        current_frame_fused_features_hwc = output_fused_tokens[:, -1, :, :, :]
        current_frame_fused_features = current_frame_fused_features_hwc.permute(0, 3, 1, 2)

        final_output_bev = self.output_proj(current_frame_fused_features)
        return final_output_bev

class OccupancyNetwork(nn.Module):
  def __init__(self,fused_bev_channels:int,
               query_point_dim:int=3,
               hidden_dim:int=128,
               output_dim:int=1):
    super().__init__()
    self.fused_bev_channels=fused_bev_channels
    self.query_point_dim=query_point_dim
    self.mlp=nn.Sequential(
        nn.Linear(fused_bev_channels+query_point_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,output_dim),
        nn.Sigmoid()
    )
  def forward(self,fused_bev_features,query_points):
    B,E,H_bev,W_bev=fused_bev_features.shape
    _,N,_=query_points.shape
    grid=query_points[:,:,:2] # Use only x,y for grid sampling
    grid=grid.unsqueeze(1) # Shape (B, 1, N, 2) for F.grid_sample

    sampled_bev_features=F.grid_sample(
        fused_bev_features,
        grid,
        mode="bilinear",
        padding_mode='zeros',
        align_corners=True
    )
    # Squeeze the height dimension (which is 1) and permute to (B, N, E)
    sampled_bev_features=sampled_bev_features.squeeze(2).permute(0,2,1)

    mlp_input=torch.cat([sampled_bev_features,query_points],dim=-1)
    return self.mlp(mlp_input)

class ObjectDetectionHead(nn.Module):
  def __init__(self,in_channels,num_classes,num_regression_params=0,hidden_dim=128):
    super().__init__()
    self.num_classes=num_classes
    self.num_regression_params=num_regression_params
    self.conv_layers=nn.Sequential(
        nn.Conv2d(in_channels,hidden_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1),
        nn.ReLU()
    )
    self.cls_head=nn.Conv2d(hidden_dim,num_classes,kernel_size=1)
    self.reg_head=nn.Conv2d(hidden_dim,num_regression_params,kernel_size=1)
  def forward(self,fused_bev_features):
    x=self.conv_layers(fused_bev_features)
    logits=self.cls_head(x)
    reg_outputs=self.reg_head(x)
    return {'logits':logits,"reg_outputs":reg_outputs}

class SemanticSegmentationHead(nn.Module):
  def __init__(self,in_channels,num_semantic_classes,bev_h=100,bev_w=100,hidden_dim=128):
    super().__init__()
    self.num_semantic_classes=num_semantic_classes
    self.bev_h=bev_h
    self.bev_w=bev_w
    self.fcn_head=nn.Sequential(
        nn.Conv2d(in_channels,hidden_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim,hidden_dim//2,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim//2,num_semantic_classes,kernel_size=1)
    )
  def forward(self,fused_bev_features:torch.Tensor)->torch.Tensor:
    logits=self.fcn_head(fused_bev_features)
    if logits.shape[2]!=self.bev_h or logits.shape[3]!=self.bev_w:
      logits=F.interpolate(logits,size=(self.bev_h,self.bev_w),mode="bilinear",align_corners=False)
    return logits

class DepthEstimationHead(nn.Module):
  def __init__(self,in_channels:int,bev_h:int=100,bev_w:int=100,hidden_dim:int=128):
    super().__init__()
    self.bev_h=bev_h
    self.bev_w=bev_w
    self.depth_head=nn.Sequential(
        nn.Conv2d(in_channels,hidden_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim,hidden_dim//2,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim//2,1,kernel_size=1)
    )
  def forward(self,fused_bev_features:torch.Tensor)->torch.Tensor:
    depth_map=self.depth_head(fused_bev_features)
    if depth_map.shape[2]!=self.bev_h or depth_map.shape[3]!=self.bev_w:
      depth_map=F.interpolate(depth_map,size=(self.bev_h,self.bev_w),mode="bilinear",align_corners=False)
    return depth_map

class PerceptionModule(nn.Module):
    def __init__(self,
                 num_cameras: int,
                 bev_h: int,
                 bev_w: int,
                 num_frames_to_fuse: int,

                 cam_img_in_channels: int = 3,
                 cam_backbone_out_channels: int = 256,
                 cam_fpn_out_channels: int = 128,
                 cam_bev_channels: int = 64,

                 lidar_voxel_in_channels: int = 4,
                 lidar_bev_channels: int = 64,

                 radar_in_channels: int = 2,
                 radar_bev_channels: int = 64,

                 fusion_embed_dim: int = 128,
                 fusion_num_heads: int = 8,
                 fusion_num_layers: int = 2,
                 fusion_dropout: float = 0.1,

                 occupancy_query_point_dim: int = 3,
                 occupancy_hidden_dim: int = 128,
                 occupancy_output_dim: int = 1,

                 det_num_classes: int = 10,
                 det_num_regression_params: int = 9,
                 det_hidden_dim: int = 128,

                 seg_num_semantic_classes: int = 5,
                 seg_hidden_dim: int = 128,

                 depth_hidden_dim: int = 128,
                 ):
        super().__init__()

        self.num_cameras = num_cameras
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_frames_to_fuse = num_frames_to_fuse

        self.camera_encoder = CameraEncoder(
            in_channels=cam_img_in_channels,
            backbone_out_channels=cam_backbone_out_channels,
            fpn_out_channels=cam_fpn_out_channels,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_channels=cam_bev_channels
        )

        self.lidar_encoder = LiDAREncoder(
            in_voxel_channels=lidar_voxel_in_channels,
            bev_channels=lidar_bev_channels,
            bev_h=bev_h,
            bev_w=bev_w
        )

        self.radar_encoder = RadarEncoder(
            in_channels=radar_in_channels,
            bev_channels=radar_bev_channels,
            bev_h=bev_h,
            bev_w=bev_w
        )

        cam_fusion_input_channels = num_cameras * cam_bev_channels
        self.fusion_module = MultiModalTemporalFusion(
            cam_bev_input_channels=cam_fusion_input_channels,
            lidar_bev_input_channels=lidar_bev_channels,
            radar_bev_input_channels=radar_bev_channels,
            bev_h=bev_h,
            bev_w=bev_w,
            fusion_embed_dim=fusion_embed_dim,
            num_frames_to_fuse=num_frames_to_fuse,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            dropout=fusion_dropout
        )

        self.occupancy_network = OccupancyNetwork(
            fused_bev_channels=fusion_embed_dim,
            query_point_dim=occupancy_query_point_dim,
            hidden_dim=occupancy_hidden_dim,
            output_dim=occupancy_output_dim
        )

        self.object_detection_head = ObjectDetectionHead(
            in_channels=fusion_embed_dim,
            num_classes=det_num_classes,
            num_regression_params=det_num_regression_params,
            hidden_dim=det_hidden_dim
        )

        self.semantic_segmentation_head = SemanticSegmentationHead(
            in_channels=fusion_embed_dim,
            num_semantic_classes=seg_num_semantic_classes,
            bev_h=bev_h,
            bev_w=bev_w,
            hidden_dim=seg_hidden_dim
        )

        self.depth_estimation_head = DepthEstimationHead(
            in_channels=fusion_embed_dim,
            bev_h=bev_h,
            bev_w=bev_w,
            hidden_dim=depth_hidden_dim
        )

    def forward(self,
                cam_input_sequence: torch.Tensor, # (B, T, num_cameras, C_img, H_img, W_img)
                lidar_input_sequence: torch.Tensor, # (B, T, C_voxel, Z_voxel, Y_voxel, X_voxel)
                radar_input_sequence: torch.Tensor, # (B, T, C_radar_raw, H_radar_raw, W_radar_raw)
                query_points: torch.Tensor # (B, N, query_point_dim) for current frame occupancy
                ) -> dict:
        batch_size, num_frames_to_fuse, _, _, _, _ = cam_input_sequence.shape

        all_cam_bev_features = []
        all_lidar_bev_features = []
        all_radar_bev_features = []

        for t in range(num_frames_to_fuse):
            current_cam_bev = self.camera_encoder(cam_input_sequence[:, t, :, :, :, :])
            all_cam_bev_features.append(current_cam_bev)

            current_lidar_bev = self.lidar_encoder(lidar_input_sequence[:, t, :, :, :, :])
            all_lidar_bev_features.append(current_lidar_bev)

            current_radar_bev = self.radar_encoder(radar_input_sequence[:, t, :, :, :])
            all_radar_bev_features.append(current_radar_bev)

        cam_bev_features_seq = torch.stack(all_cam_bev_features, dim=1)
        lidar_bev_features_seq = torch.stack(all_lidar_bev_features, dim=1)
        radar_bev_features_seq = torch.stack(all_radar_bev_features, dim=1)

        # Fused BEV features for the current (last) frame, using temporal context
        fused_bev_features_current_frame = self.fusion_module(
            cam_bev_features_seq,
            lidar_bev_features_seq,
            radar_bev_features_seq
        )

        object_detections = self.object_detection_head(fused_bev_features_current_frame)
        semantic_segmentation = self.semantic_segmentation_head(fused_bev_features_current_frame)
        depth_map = self.depth_estimation_head(fused_bev_features_current_frame)
        occupancy_probabilities = self.occupancy_network(fused_bev_features_current_frame, query_points)

        return {
            "fused_bev_features": fused_bev_features_current_frame,
            "object_detections": object_detections,
            "semantic_segmentation": semantic_segmentation,
            "depth_map": depth_map,
            "occupancy_probabilities": occupancy_probabilities
        }

class AgentFeatureExtractor(nn.Module):
  def __init__(self, fused_bev_channels: int, agent_input_features_dim: int, agent_bev_patch_size: int, output_feature_dim: int):
    super().__init__()
    self.fused_bev_channels = fused_bev_channels
    self.agent_input_features_dim = agent_input_features_dim
    self.agent_bev_patch_size = agent_bev_patch_size
    self.output_feature_dim = output_feature_dim
    combined_input_dim = agent_input_features_dim + (fused_bev_channels * agent_bev_patch_size * agent_bev_patch_size)
    self.projection_mlp = nn.Sequential(
        nn.Linear(combined_input_dim, output_feature_dim * 2),
        nn.ReLU(),
        nn.Linear(output_feature_dim * 2, output_feature_dim)
    )

  def forward(self, fused_bev_features: torch.Tensor, detected_agents_states: torch.Tensor) -> torch.Tensor:
    B, C_fused, H_bev, W_bev = fused_bev_features.shape
    _, N_agents, C_agent_input = detected_agents_states.shape
    agent_features_list = []

    # Ensure patch size is odd to have a clear center, or handle even sizes
    # For simplicity, if patch_size is even, consider it one less to be odd for centering.
    effective_patch_size = self.agent_bev_patch_size if self.agent_bev_patch_size % 2 != 0 else self.agent_bev_patch_size - 1
    if effective_patch_size < 1: effective_patch_size = 1 # Ensure at least 1x1 patch

    # Generate grid coordinates relative to the agent's center in normalized [-1, 1] space
    half_patch_extent = effective_patch_size / max(H_bev, W_bev) * (W_bev / 2)

    for b in range(B):
      batch_agent_features = []
      for n in range(N_agents):
        agent_numerical_attrs = detected_agents_states[b, n, :]
        agent_x_norm = agent_numerical_attrs[0] # Assumed normalized x
        agent_y_norm = agent_numerical_attrs[1] # Assumed normalized y

        # Create a grid for sampling around the agent's normalized (x,y) coordinates
        xs = torch.linspace(-half_patch_extent, half_patch_extent, effective_patch_size, device=fused_bev_features.device)
        ys = torch.linspace(-half_patch_extent, half_patch_extent, effective_patch_size, device=fused_bev_features.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

        # Offset grid to agent's position
        grid_x_abs = agent_x_norm + grid_x
        grid_y_abs = agent_y_norm + grid_y

        grid = torch.stack((grid_x_abs, grid_y_abs), dim=-1).unsqueeze(0) # (1, H_patch, W_patch, 2)

        # Sample the BEV features
        # fused_bev_features[b:b+1] to keep batch dimension for grid_sample
        sampled_bev_patch = F.grid_sample(
            fused_bev_features[b:b+1], # (1, C, H, W)
            grid, # (1, H_patch, W_patch, 2)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ) # Output: (1, C, H_patch, W_patch)

        flattened_bev_patch = sampled_bev_patch.flatten() # (C * H_patch * W_patch)

        combined_agent_features = torch.cat((agent_numerical_attrs, flattened_bev_patch), dim=-1)
        batch_agent_features.append(combined_agent_features)

      if len(batch_agent_features) > 0:
        agent_features_list.append(torch.stack(batch_agent_features)) # (N_agents, combined_input_dim)
      else:
        # Handle case with no agents (e.g., return zeros or an empty tensor)
        agent_features_list.append(torch.zeros(
            N_agents, self.agent_input_features_dim + (self.fused_bev_channels * effective_patch_size * effective_patch_size),
            device=fused_bev_features.device
        ))

    # Stack all batch results: (B, N_agents, combined_input_dim)
    return self.projection_mlp(torch.stack(agent_features_list))

class TemporalEncoder(nn.Module):
  def __init__(self,
               agent_feature_dim:int,
               temporal_embed_dim:int,
               max_frames:int,
               num_temporal_heads:int=8,
               num_temporal_layers:int=2,
               dropout:float=0.1):
    super().__init__()
    self.agent_feature_dim=agent_feature_dim
    self.temporal_embed_dim=temporal_embed_dim
    self.max_frames=max_frames
    if agent_feature_dim!=temporal_embed_dim:
      self.input_projection=nn.Linear(agent_feature_dim,temporal_embed_dim)
    else:
      self.input_projection=nn.Identity()
    encoder_layer=nn.TransformerEncoderLayer(
      d_model=temporal_embed_dim,
      nhead=num_temporal_heads,
      dim_feedforward=temporal_embed_dim*2,
      dropout=dropout,
      batch_first=True
    )
    self.temporal_transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_temporal_layers)
    self.temporal_pos_embedding=nn.Parameter(torch.randn(1,max_frames,temporal_embed_dim))
  def forward(self,agent_features_seq:torch.Tensor)->torch.Tensor:
    B,T,N_agents,F=agent_features_seq.shape
    assert T<=self.max_frames, f"Input sequence length {T} exceeds max_frames {self.max_frames}"
    # Reshape (B, T, N_agents, F) to (B*N_agents, T, F) for transformer
    input_for_transformer=agent_features_seq.view(B*N_agents,T,F)
    projected_features=self.input_projection(input_for_transformer)
    pos_embed=self.temporal_pos_embedding[:,:T,:] # Slice positional embeddings to match current sequence length
    features_with_pe=projected_features+pos_embed
    transformer_output=self.temporal_transformer(features_with_pe)
    # Take the output corresponding to the last time step (current frame)
    context_aware_features=transformer_output[:,-1,:]
    # Reshape back to (B, N_agents, temporal_embed_dim)
    return context_aware_features.view(B,N_agents,self.temporal_embed_dim)

class InteractionModel(nn.Module):
  def __init__(self, agent_feature_dim: int,
               scene_context_feature_dim: int,
               interaction_embed_dim: int,
               num_attention_heads: int = 8,
               num_transformer_layers: int = 2,
               dropout: float = 0.1,
               max_agents: int = 20):
    super().__init__()
    self.interaction_embed_dim = interaction_embed_dim
    self.max_agents = max_agents

    self.agent_feature_projection = nn.Linear(agent_feature_dim, interaction_embed_dim)
    self.scene_context_projection = nn.Linear(scene_context_feature_dim, interaction_embed_dim)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=interaction_embed_dim,
        nhead=num_attention_heads,
        dim_feedforward=interaction_embed_dim * 2,
        dropout=dropout,
        batch_first=True # Input and output tensors are (batch, sequence, feature)
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    # Positional embeddings for agents and a single scene token
    self.agent_pos_embedding = nn.Parameter(torch.randn(1, max_agents, interaction_embed_dim))
    self.scene_pos_embedding = nn.Parameter(torch.randn(1, 1, interaction_embed_dim))

  def forward(self, temporal_encoded_agent_features: torch.Tensor, scene_context_features: torch.Tensor) -> torch.Tensor:
    B, N_agents, F_agent = temporal_encoded_agent_features.shape
    #_, F_context, H_bev, W_bev = scene_context_features.shape # F_context should be scene_context_feature_dim

    assert N_agents <= self.max_agents, f"Number of agents {N_agents} exceeds max_agents {self.max_agents}"

    # Project agent features to common embedding dimension
    agent_tokens = self.agent_feature_projection(temporal_encoded_agent_features)

    # Pool scene context to get a single global scene token per batch
    # scene_context_features: (B, F_context, H_bev, W_bev)
    global_scene_pooled = F.adaptive_avg_pool2d(scene_context_features, (1, 1)).squeeze(-1).squeeze(-1)
    scene_token = self.scene_context_projection(global_scene_pooled) # (B, interaction_embed_dim)
    scene_token = scene_token.unsqueeze(1) # (B, 1, interaction_embed_dim)

    # Add positional embeddings
    agent_pos_embed = self.agent_pos_embedding[:, :N_agents, :]

    # Concatenate agent tokens and scene token, adding positional embeddings
    # Ensure agents and scene tokens have positional embeddings before concatenation for transformer
    tokens_with_pe = torch.cat([
        agent_tokens + agent_pos_embed,
        scene_token + self.scene_pos_embedding # temporal_pos_embedding is not defined here
    ], dim=1)

    # Apply Transformer Encoder
    transformer_output = self.transformer_encoder(tokens_with_pe)

    # Extract only agent-specific outputs
    # The first N_agents tokens in the output correspond to the input agent tokens
    interaction_aware_features = transformer_output[:, :N_agents, :]
    return interaction_aware_features

class TrajectoryDecoder(nn.Module):
  def __init__(self,agent_feature_dim:int,
               num_future_steps:int,num_modes:int,
               trajectory_point_dim:int,hidden_dim:int=256):
    super().__init__()
    self.num_future_steps=num_future_steps
    self.num_modes=num_modes
    self.trajectory_point_dim=trajectory_point_dim
    output_size_per_mode=(num_future_steps*trajectory_point_dim)+1 # +1 for confidence score
    total_output_size=num_modes*output_size_per_mode

    self.mlp=nn.Sequential(
        nn.Linear(agent_feature_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,total_output_size)
    )

  def forward(self,interaction_aware_agent_features:torch.Tensor)->dict:
    B,N_agents,F_interaction=interaction_aware_agent_features.shape
    raw_predictions=self.mlp(interaction_aware_agent_features)

    output_size_per_mode=(self.num_future_steps*self.trajectory_point_dim)+1

    # Reshape predictions to separate modes and their components
    reshaped_predictions=raw_predictions.view(B,N_agents,self.num_modes,output_size_per_mode)

    # Extract flattened trajectories and mode confidences
    predicted_trajectories_flat=reshaped_predictions[:,:,:,:-1]
    mode_confidences=reshaped_predictions[:,:,:,-1]

    # Reshape flattened trajectories to (B, N_agents, num_modes, num_future_steps, trajectory_point_dim)
    predicted_trajectories=predicted_trajectories_flat.view(
        B,N_agents,self.num_modes,self.num_future_steps,self.trajectory_point_dim
    )

    # Apply softmax to mode confidences to get probabilities
    mode_probabilities=F.softmax(mode_confidences,dim=-1)

    return {
            'predicted_trajectories': predicted_trajectories,
            'mode_probabilities': mode_probabilities
        }

class PredictionModule(nn.Module):
    def __init__(self, fused_bev_channels: int, agent_input_features_dim: int, agent_bev_patch_size: int,
                 agent_feature_extractor_output_dim: int, num_frames_to_fuse: int, temporal_encoder_max_frames: int,
                 temporal_encoder_embed_dim: int, interaction_model_embed_dim: int, interaction_model_max_agents: int,
                 num_future_steps: int, num_modes: int, trajectory_point_dim: int,
                 temporal_encoder_num_heads: int = 8, temporal_encoder_num_layers: int = 2,
                 temporal_encoder_dropout: float = 0.1, interaction_model_num_heads: int = 8,
                 interaction_model_num_layers: int = 2, interaction_model_dropout: float = 0.1,
                 trajectory_decoder_hidden_dim: int = 256):
        super().__init__()

        self.num_frames_to_fuse = num_frames_to_fuse
        self.agent_input_features_dim = agent_input_features_dim

        self.agent_feature_extractor = AgentFeatureExtractor(
            fused_bev_channels=fused_bev_channels,
            agent_input_features_dim=agent_input_features_dim,
            agent_bev_patch_size=agent_bev_patch_size,
            output_feature_dim=agent_feature_extractor_output_dim
        )

        self.temporal_encoder = TemporalEncoder(
            agent_feature_dim=agent_feature_extractor_output_dim,
            temporal_embed_dim=temporal_encoder_embed_dim,
            max_frames=temporal_encoder_max_frames,
            num_temporal_heads=temporal_encoder_num_heads,
            num_temporal_layers=temporal_encoder_num_layers,
            dropout=temporal_encoder_dropout
        )

        self.interaction_model = InteractionModel(
            agent_feature_dim=temporal_encoder_embed_dim,
            scene_context_feature_dim=fused_bev_channels,
            interaction_embed_dim=interaction_model_embed_dim,
            max_agents=interaction_model_max_agents,
            num_attention_heads=interaction_model_num_heads,
            num_transformer_layers=interaction_model_num_layers,
            dropout=interaction_model_dropout
        )

        self.trajectory_decoder = TrajectoryDecoder(
            agent_feature_dim=interaction_model_embed_dim,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            trajectory_point_dim=trajectory_point_dim,
            hidden_dim=trajectory_decoder_hidden_dim
        )

    def forward(self, fused_bev_features_seq: torch.Tensor, detected_agents_states_seq: torch.Tensor) -> dict:
        B, T, C_fused, H_bev, W_bev = fused_bev_features_seq.shape
        _, _, N_agents, F_agent_input = detected_agents_states_seq.shape

        assert T == self.num_frames_to_fuse, f"Input sequence length {T} must match num_frames_to_fuse {self.num_frames_to_fuse}"

        agent_centric_features_seq = []
        for t in range(T):
            current_fused_bev = fused_bev_features_seq[:, t, :, :, :]
            current_detected_agents_states = detected_agents_states_seq[:, t, :, :]

            agent_centric_features = self.agent_feature_extractor(
                current_fused_bev,
                current_detected_agents_states
            )
            agent_centric_features_seq.append(agent_centric_features)

        agent_centric_features_seq_tensor = torch.stack(agent_centric_features_seq, dim=1)

        temporal_encoded_agent_features = self.temporal_encoder(
            agent_centric_features_seq_tensor
        )

        current_frame_fused_bev = fused_bev_features_seq[:, -1, :, :, :]
        interaction_aware_agent_features = self.interaction_model(
            temporal_encoded_agent_features,
            current_frame_fused_bev
        )

        prediction_outputs = self.trajectory_decoder(
            interaction_aware_agent_features
        )

        return prediction_outputs

class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class DecisionMakingNetwork(nn.Module):
  def __init__(self,input_dim:int,num_high_level_actions:int,hidden_dim:int=128):
    super().__init__()
    self.mlp=nn.Sequential(
      nn.Linear(input_dim,hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim,num_high_level_actions)
    )
  def forward(self,planning_context:torch.Tensor)->torch.Tensor:
    return self.mlp(planning_context)

class TrajectoryGenerator(nn.Module):
  def __init__(self,input_dim:int,num_candidate_trajectories:int,num_future_steps:int,trajectory_point_dim:int,hidden_dim:int=256):
    super().__init__()
    self.num_candidate_trajectories=num_candidate_trajectories
    self.num_future_steps=num_future_steps
    self.trajectory_point_dim=trajectory_point_dim
    output_size_per_trajectory=num_future_steps*trajectory_point_dim
    total_output_size=num_candidate_trajectories*output_size_per_trajectory
    self.mlp=nn.Sequential(
        nn.Linear(input_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,total_output_size)
    )
  def forward(self,planning_context:torch.Tensor,high_level_action:torch.Tensor)->torch.Tensor:
    combined_input=torch.cat([planning_context,high_level_action],dim=-1)
    raw_trajectories=self.mlp(combined_input)
    trajectories=raw_trajectories.view(
        raw_trajectories.shape[0],
        self.num_candidate_trajectories,
        self.num_future_steps,
        self.trajectory_point_dim
    )
    return trajectories

class TrajectoryEvaluator(nn.Module):
  def __init__(self,trajectory_feature_dim:int,hidden_dim:int=128):
    super().__init__()
    self.mlp=nn.Sequential(
        nn.Linear(trajectory_feature_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,1)
    )
  def forward(self,candidate_trajectories:torch.Tensor,planning_context:torch.Tensor)->torch.Tensor:
    B,N_candidates,N_steps,N_dims=candidate_trajectories.shape
    flattened_trajectories=candidate_trajectories.view(B,N_candidates,-1)
    expanded_context=planning_context.unsqueeze(1).expand(-1,N_candidates,-1)
    combined_features=torch.cat([flattened_trajectories,expanded_context],dim=-1)
    scores=self.mlp(combined_features)
    return scores.squeeze(-1)

class PlanningModule(nn.Module):
  def __init__(self, context_encoder_input_dim: int, context_embedding_dim: int, num_high_level_actions: int,
               num_candidate_trajectories: int, num_future_steps: int, trajectory_point_dim: int,
               hidden_dim: int = 256):
    super().__init__()
    self.num_high_level_actions = num_high_level_actions
    self.num_candidate_trajectories = num_candidate_trajectories

    self.context_encoder = ContextEncoder(context_encoder_input_dim, context_embedding_dim, hidden_dim=hidden_dim)
    self.decision_making_network = DecisionMakingNetwork(context_embedding_dim, num_high_level_actions, hidden_dim=hidden_dim)

    trajectory_generator_input_dim = context_embedding_dim + num_high_level_actions
    self.trajectory_generator = TrajectoryGenerator(
        trajectory_generator_input_dim,
        num_candidate_trajectories,
        num_future_steps,
        trajectory_point_dim,
        hidden_dim=hidden_dim
    )

    trajectory_feature_dim = (num_future_steps * trajectory_point_dim) + context_embedding_dim
    self.trajectory_evaluator = TrajectoryEvaluator(trajectory_feature_dim, hidden_dim=hidden_dim)

  def forward(self, fused_bev_features: torch.Tensor, occupancy_probabilities: torch.Tensor,
              semantic_segmentation: torch.Tensor, predicted_trajectories: torch.Tensor,
              mode_probabilities: torch.Tensor, ego_vehicle_state: torch.Tensor) -> dict:
    B = ego_vehicle_state.shape[0]

    fused_bev_flat = fused_bev_features.flatten(start_dim=1)
    occupancy_flat = occupancy_probabilities.flatten(start_dim=1)
    semantic_flat = semantic_segmentation.flatten(start_dim=1)
    predicted_trajectories_flat = predicted_trajectories.flatten(start_dim=1)
    mode_probabilities_flat = mode_probabilities.flatten(start_dim=1)

    context_encoder_input = torch.cat([
        fused_bev_flat,
        occupancy_flat,
        semantic_flat,
        predicted_trajectories_flat,
        mode_probabilities_flat,
        ego_vehicle_state
    ], dim=-1)

    planning_context = self.context_encoder(context_encoder_input)

    high_level_action_logits = self.decision_making_network(planning_context)
    high_level_action_idx = torch.argmax(high_level_action_logits, dim=-1)
    high_level_action_one_hot = F.one_hot(high_level_action_idx, num_classes=self.num_high_level_actions).float()

    candidate_trajectories = self.trajectory_generator(planning_context, high_level_action_one_hot)

    scores = self.trajectory_evaluator(candidate_trajectories, planning_context)
    best_trajectory_idx = torch.argmax(scores, dim=-1)

    # Select the best trajectory for each batch element
    # Use gather or advanced indexing to pick the best trajectory directly
    selected_trajectory = candidate_trajectories[torch.arange(B), best_trajectory_idx, :, :]

    return {
            "high_level_action_logits": high_level_action_logits,
            "high_level_action_index": high_level_action_idx,
            "candidate_trajectories": candidate_trajectories,
            "trajectory_scores": scores,
            "selected_trajectory": selected_trajectory
        }

# ----------------- End of Helper Modules -----------------
