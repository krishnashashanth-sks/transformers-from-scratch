import torch.nn as nn
import torch
import torch.nn.functional as F

# --- 2D CNN Backbone Components (ResNet-like) ---
class BasicBlock(nn.Module):
  expansion=1
  def __init__(self,in_channels,out_channels,stride=1):
    super(BasicBlock,self).__init__()
    self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
    self.bn1=nn.BatchNorm2d(out_channels)
    self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
    self.bn2=nn.BatchNorm2d(out_channels)
    self.shortcut=nn.Sequential()
    if stride !=1 or in_channels!=self.expansion*out_channels:
      self.shortcut=nn.Sequential(
          nn.Conv2d(in_channels,self.expansion*out_channels,kernel_size=1,stride=stride,bias=False),
          nn.BatchNorm2d(self.expansion*out_channels)
      )
  def forward(self,x):
    out=F.relu(self.bn1(self.conv1(x)))
    out=self.bn2(self.conv2(out))
    out+=self.shortcut(x)
    return F.relu(out)

class Bottleneck(nn.Module):
  expansion=4
  def __init__(self,in_channels,out_channels,stride=1):
    super(Bottleneck,self).__init__()
    self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
    self.bn1=nn.BatchNorm2d(out_channels)
    self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
    self.bn2=nn.BatchNorm2d(out_channels)
    self.conv3=nn.Conv2d(out_channels,self.expansion*out_channels,kernel_size=1,bias=False)
    self.bn3=nn.BatchNorm2d(self.expansion*out_channels)
    self.shortcut=nn.Sequential()
    if stride !=1 or in_channels !=self.expansion*out_channels:
      self.shortcut=nn.Sequential(
          nn.Conv2d(in_channels,self.expansion*out_channels,kernel_size=1,stride=stride,bias=False),
          nn.BatchNorm2d(self.expansion*out_channels)
      )
  def forward(self,x):
    out=F.relu(self.bn1(self.conv1(x)))
    out=F.relu(self.bn2(self.conv2(out)))
    out=self.bn3(self.conv3(out))
    out+=self.shortcut(x)
    return F.relu(out)

class CameraBackbone(nn.Module):
      def __init__(self,block=BasicBlock,num_blocks=[2,2,2,2],in_channels=3,output_channels=512):
        super(CameraBackbone, self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2=self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3=self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4=self._make_layer(block,512,num_blocks[3],stride=2)
        
        # Adjust final output channels if different from block.expansion*512
        if block.expansion * 512 != output_channels:
          self.final_conv = nn.Conv2d(block.expansion * 512, output_channels, kernel_size=1, bias=False)
        else:
          self.final_conv = nn.Identity()

      def _make_layer(self,block,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for current_stride in strides:
          layers.append(block(self.in_channels,out_channels,current_stride))
          self.in_channels=out_channels*block.expansion
        return nn.Sequential(*layers)
      def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.maxpool(out)
        # Note: Original implementation only passed through layer4 directly, but typically ResNet goes through all layers sequentially
        # For ResNet-style feature extraction, often you take output from intermediate layers or the final global pool.
        # Assuming the intent is to get features from the last layer:
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        return self.final_conv(out)


# --- View Transformer ---
class ViewTransformer(nn.Module):
  def __init__(self,img_feat_dim,bev_h,bev_w,bev_c,depth_channels=64,num_cameras=6):
    super(ViewTransformer,self).__init__()
    self.img_feat_dim=img_feat_dim
    self.bev_h=bev_h
    self.bev_w=bev_w
    self.bev_c=bev_c
    self.depth_channels=depth_channels
    self.num_cameras=num_cameras

    # Initialize depth values
    self.depth_values = torch.linspace(1.0, 60.0, self.depth_channels)

    self.depth_net=nn.Sequential(
        nn.Conv2d(self.img_feat_dim,self.img_feat_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(self.img_feat_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.img_feat_dim,self.depth_channels,kernel_size=1)
    )
    self.output_conv=nn.Conv2d(img_feat_dim,bev_c,kernel_size=1)

  def forward(self,img_features,cam_intrinsics,cam_extrinsics):
    batch_size=img_features.shape[0]//self.num_cameras
    H_img,W_img=img_features.shape[-2:]
    device=img_features.device

    # Reshape img_features for easier processing per camera
    imag_features_reshaped=img_features.view(batch_size,self.num_cameras,self.img_feat_dim,H_img,W_img)

    depth_logits=self.depth_net(img_features)
    depth_probs=F.softmax(depth_logits,dim=1)
    depth_probs_reshaped=depth_probs.view(batch_size,self.num_cameras,self.depth_channels,H_img,W_img)

    # Apply depth probabilities to image features
    weighted_features = imag_features_reshaped.unsqueeze(3) * depth_probs_reshaped.unsqueeze(2)
    # weighted_features shape: (B, N_cam, C_img, D, H_img, W_img)

    # Generate 2D image coordinates (u, v)
    u_coords=torch.arange(W_img,device=device,dtype=torch.float32)
    v_coords=torch.arange(H_img,device=device,dtype=torch.float32)
    grid_u,grid_v=torch.meshgrid(u_coords,v_coords,indexing='xy')
    image_coords=torch.stack([grid_u,grid_v,torch.ones_like(grid_u)],dim=-1)
    image_coords=image_coords.permute(2,0,1) # Shape: (3, H_img, W_img)
    image_coords=image_coords.view(3,-1).unsqueeze(0) # Shape: (1, 3, H_img*W_img)

    # Expand depth values to match dimensions for multiplication
    expanded_depth_values=self.depth_values.view(1,self.depth_channels,1,1).to(device) # Shape: (1,D,1,1)

    # Inverse camera intrinsics for projection
    inv_cam_intrinsics=torch.inverse(cam_intrinsics) # (B*N, 3, 3)
    inv_cam_extrinsics = torch.inverse(cam_extrinsics) # (B*N, 4, 4)

    # Project 2D image points to 3D camera coordinates
    # cam_points_flat: (B*N, 3, H_img*W_img)
    cam_points_flat=torch.matmul(inv_cam_intrinsics,image_coords)

    # Expand cam_points_flat for depth dimension and multiply by expanded_depth_values
    # cam_points_for_depth_mult: (B*N, 1, 3, H_img*W_img)
    cam_points_for_depth_mult = cam_points_flat.unsqueeze(1) 

    cam_3d_points_in_camera_frame = cam_points_for_depth_mult * expanded_depth_values

    # Permute to (B*N, D, H_img*W_img, 3) for consistency with world_3d_points
    cam_3d_points_in_camera_frame = cam_3d_points_in_camera_frame.permute(0, 1, 3, 2)

    # Convert to homogeneous coordinates and transform to world coordinates
    # cam_3d_points_homogeneous: (B*N, D, H_img*W_img, 4)
    cam_3d_points_homogeneous=F.pad(cam_3d_points_in_camera_frame,(0,1),'constant',1)

    # Reshape for efficient matrix multiplication
    # cam_3d_points_homogeneous_flat: (B*N, D*HW, 4)
    cam_3d_points_homogeneous_flat = cam_3d_points_homogeneous.view(
        batch_size * self.num_cameras, self.depth_channels * H_img * W_img, 4
    )
    
    # world_3d_points_homogeneous_result: (B*N, D*HW, 4, 1)
    # inv_cam_extrinsics: (B*N, 4, 4)
    world_3d_points_homogeneous_result = torch.matmul(
        inv_cam_extrinsics.unsqueeze(1), # (B*N, 1, 4, 4) - will broadcast
        cam_3d_points_homogeneous_flat.unsqueeze(-1) # (B*N, D*HW, 4, 1)
    )
    # world_3d_points: (B*N, D*HW, 3)
    world_3d_points = world_3d_points_homogeneous_result.squeeze(-1)[..., :3]

    # Reshape weighted features to match world_3d_points for scattering
    # weighted_features: (batch_size, num_cameras, img_feat_dim, depth_channels, H_img, W_img)
    # weighted_features_lifted: (B*N, D*HW, C_img)
    weighted_features_lifted = weighted_features.permute(0,1,3,4,5,2).reshape(
        batch_size * self.num_cameras, self.depth_channels * H_img * W_img, self.img_feat_dim
    )

    # Flatten world_3d_points and weighted_features for scattering
    world_3d_points_all=world_3d_points.view(-1,3) # (B*N*D*HW, 3)
    weighted_features_all=weighted_features_lifted.reshape(-1,self.img_feat_dim) # (B*N*D*HW, C_img)

    # Define BEV grid boundaries (example values)
    bev_x_min=-50.0
    bev_x_max=50.0
    bev_y_min=-50.0
    bev_y_max=50.0

    # Project world coordinates to BEV pixel coordinates
    x_world=world_3d_points_all[:,0]
    y_world=world_3d_points_all[:,1]

    # Convert world coordinates to pixel coordinates, then round to integer indices
    x_pixel=(x_world-bev_x_min)/((bev_x_max-bev_x_min)/self.bev_w)
    y_pixel=(y_world-bev_y_min)/((bev_y_max-bev_y_min)/self.bev_h)
    x_bev=torch.round(x_pixel).long()
    y_bev=torch.round(y_pixel).long()

    # Create a mask for valid BEV coordinates
    valid_mask=(x_bev>=0)&(x_bev<self.bev_w)&(y_bev>=0)&(y_bev<self.bev_h)

    x_bev_valid=x_bev[valid_mask]
    y_bev_valid=y_bev[valid_mask]
    weighted_features_valid=weighted_features_all[valid_mask]

    # Create batch indices for scatter_add_
    points_per_camera = self.depth_channels * H_img * W_img
    # The original cam_intrinsics/extrinsics are (B*N, ...), so the world_3d_points_all is also flattened across B*N
    # We need to recover the batch index for each point in world_3d_points_all
    camera_indices = torch.arange(batch_size * self.num_cameras, device=device).repeat_interleave(points_per_camera)
    batch_indices_for_points = camera_indices // self.num_cameras
    batch_indices_valid = batch_indices_for_points[valid_mask]

    # Create flat indices for the BEV map for scatter_add_
    # linear_indices = batch_idx * (bev_h * bev_w) + y_idx * bev_w + x_idx
    spatial_indices_valid = batch_indices_valid * (self.bev_h * self.bev_w) + y_bev_valid * self.bev_w + x_bev_valid

    # Create an output tensor to scatter into
    bev_features_flat = torch.zeros(batch_size * self.bev_h * self.bev_w, self.img_feat_dim, device=device)

    # Scatter weighted features to the BEV grid positions
    # scatter_add_ expects index to be a LongTensor of same size as src, or broadcastable if src has extra dimensions.
    # For dim=0, index shape should be (N_valid,)
    bev_features_flat.scatter_add_(0, spatial_indices_valid.unsqueeze(1).expand(-1, self.img_feat_dim), weighted_features_valid)

    # Reshape back to (B, C_img, H, W)
    bev_features_final=bev_features_flat.view(batch_size,self.bev_h,self.bev_w,self.img_feat_dim).permute(0,3,1,2)

    # Apply final convolution
    return self.output_conv(bev_features_final)


# --- LiDAR Stream Components ---
class VoxelFeatureEncoder(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(VoxelFeatureEncoder,self).__init__() # Corrected super call
    self.mlp=nn.Sequential(
        nn.Linear(in_channels,out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Linear(out_channels,out_channels),
        nn.BatchNorm1d(out_channels), # Corrected BatchNorm1d typo
        nn.ReLU(inplace=True)
    )
  def forward(self, voxel_points):
    # voxel_points: (num_voxels, max_points_per_voxel, in_channels)
    num_voxels, max_points_per_voxel, in_channels = voxel_points.shape
    
    # --- FIX: Early exit if there are no valid voxels ---
    if num_voxels == 0:
        # Dynamically fetch the out_channels from the final MLP layer
        out_channels = self.mlp[-2].out_features if hasattr(self.mlp[-2], 'out_features') else 64
        return torch.zeros(0, out_channels, device=voxel_points.device, dtype=voxel_points.dtype)
    # ----------------------------------------------------

    # Reshape for MLP: (num_voxels * max_points_per_voxel, in_channels)
    reshaped_points = voxel_points.view(-1, in_channels)
    
    # point_wise_features: (num_voxels * max_points_per_voxel, out_channels)
    point_wise_features = self.mlp(reshaped_points)
    
    # Reshape back to (num_voxels, max_points_per_voxel, out_channels)
    point_wise_features = point_wise_features.view(num_voxels, max_points_per_voxel, -1)
    
    # Max-pooling across points within each voxel: (num_voxels, out_channels)
    voxel_features, _ = torch.max(point_wise_features, dim=1) 
    return voxel_features

class VoxelToBEVConverter(nn.Module):
  def __init__(self,voxel_feature_channels:int,grid_size:tuple,bev_channels:int):
    super(VoxelToBEVConverter,self).__init__()
    self.voxel_feature_channels=voxel_feature_channels
    self.grid_size=grid_size # (X_DIM, Y_DIM, Z_DIM)
    self.bev_channels=bev_channels

    if voxel_feature_channels!=bev_channels:
      self.feature_projection=nn.Linear(voxel_feature_channels,bev_channels)
    else:
      self.feature_projection=nn.Identity()

  def forward(self,voxel_features:torch.Tensor,voxel_coords:torch.Tensor)->torch.Tensor:
    # voxel_features: (N_valid_voxels, voxel_feature_channels)
    # voxel_coords: (N_valid_voxels, 4) -> [batch_idx, x_idx, y_idx, z_idx]
    
    device=voxel_features.device

    # Handle empty input case (no valid voxels)
    if voxel_features.shape[0]==0:
      # If voxel_coords is empty, cannot infer batch_size from it, assume 1.
      batch_size = int(voxel_coords[:,0].max().item())+1 if voxel_coords.numel()>0 else 1
      return torch.zeros( # Corrected torxh to torch
          batch_size,self.bev_channels,self.grid_size[1],self.grid_size[0],
          device=device
      )
    
    batch_idx=voxel_coords[:,0]
    x_idx=voxel_coords[:,1]
    y_idx=voxel_coords[:,2]
    # z_idx = voxel_coords[:,3] # Not used for 2D BEV projection

    batch_size=int(batch_idx.max().item())+1
    processed_voxel_features=self.feature_projection(voxel_features)

    # Calculate linear indices for scattering into a flattened BEV map
    # Each batch has a separate BEV grid. Flattened index: batch_idx * (H*W) + y_idx * W + x_idx
    linear_indices=batch_idx * (self.grid_size[0]*self.grid_size[1]) + \
                   y_idx * self.grid_size[0] + x_idx # grid_size[0] is W, grid_size[1] is H

    # Initialize a flat BEV feature tensor to scatter into
    bev_features_flat=torch.zeros(
        batch_size*self.grid_size[0]*self.grid_size[1],
        self.bev_channels,
        device=device,
        dtype=processed_voxel_features.dtype
    )

    # Scatter features. `scatter_add_` needs `index` to be the same size as `src` in `dim`
    bev_features_flat.scatter_add_(
        0, 
        linear_indices.unsqueeze(1).expand(-1, self.bev_channels), 
        processed_voxel_features
    )

    # Reshape the flattened BEV features back to (B, C, H, W)
    # Note: grid_size is (X_DIM, Y_DIM, Z_DIM) so grid_size[0] is width, grid_size[1] is height
    bev_features=bev_features_flat.view(
        batch_size, self.grid_size[1], self.grid_size[0], self.bev_channels
    ).permute(0,3,1,2) # Permute to (B, C, H, W)
    
    return bev_features

class LiDAREncoder(nn.Module):
  def __init__(self,voxel_in_channels:int,
               voxel_feature_out_channels:int,
               bev_channels:int,
               grid_size:tuple):
    super(LiDAREncoder,self).__init__()
    self.vfe=VoxelFeatureEncoder(
        in_channels=voxel_in_channels, # Corrected to voxel_in_channels
        out_channels=voxel_feature_out_channels
    )
    self.voxel_to_bev=VoxelToBEVConverter(
        voxel_feature_channels=voxel_feature_out_channels, # Use output channels of VFE
        grid_size=grid_size,
        bev_channels=bev_channels
    )

  def forward(self,voxel_points:torch.Tensor,voxel_coords:torch.Tensor)->torch.Tensor:
    # voxel_points: (num_voxels_total, max_points_per_voxel, in_channels)
    # voxel_coords: (num_voxels_total, 4) -> [batch_idx, x_idx, y_idx, z_idx]
    
    # 1. Voxel Feature Encoding
    voxel_features=self.vfe(voxel_points) # (num_voxels_total, voxel_feature_out_channels)
    
    # 2. Voxel-to-BEV Projection
    bev_feature_map=self.voxel_to_bev(voxel_features,voxel_coords) # (B, bev_channels, H, W)
    
    return bev_feature_map
