import torch
import torch.nn as nn
from layers import CameraBackbone, ViewTransformer, LiDAREncoder

class BEVFusionModel(nn.Module):
    def __init__(self, bev_h=200, bev_w=200, camera_feat_dim=512, lidar_feat_dim=256, 
                 combined_feat_dim=256, num_cameras=6, depth_channels=64, voxel_size=(0.5, 0.5, 0.2)):
        super(BEVFusionModel, self).__init__()
        self.num_cameras = num_cameras
        self.bev_h = bev_h
        self.bev_w = bev_w

        # 1. Camera Stream Components
        self.camera_backbone = CameraBackbone(output_channels=camera_feat_dim)
        self.view_transformer = ViewTransformer(
            img_feat_dim=camera_feat_dim,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_c=combined_feat_dim,
            depth_channels=depth_channels,
            num_cameras=num_cameras
        )

        # 2. LiDAR Stream Components
        # Derived dimensions based on BEV range [-50, -50, -5, 50, 50, 3] and calculated voxel sizes
        grid_x_dim = bev_w
        grid_y_dim = bev_h
        grid_z_dim = int((3.0 - (-5.0)) / voxel_size[2]) # e.g., (8.0 / 0.2) = 40
        self.grid_size = (grid_x_dim, grid_y_dim, grid_z_dim)

        self.lidar_encoder = LiDAREncoder(
            voxel_in_channels=4, # [x, y, z, intensity]
            voxel_feature_out_channels=64,
            bev_channels=combined_feat_dim,
            grid_size=self.grid_size
        )

        # 3. Fusion Layer
        # Concatenates camera BEV features and LiDAR BEV features along the channel axis
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(combined_feat_dim * 2, combined_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(combined_feat_dim),
            nn.ReLU(inplace=True)
        )

        # 4. Detection Head (Outputs 3 channels for dummy heatmap targets)
        self.detection_head = nn.Sequential(
            nn.Conv2d(combined_feat_dim, combined_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(combined_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(combined_feat_dim, 3, kernel_size=1) 
        )

    def forward(self, camera_images, cam_intrinsics, cam_extrinsics, lidar_voxel_points, lidar_voxel_coords):
        """
        Args:
            camera_images: (B, N_cam, C, H, W) -> [B, 6, 3, 224, 224]
            cam_intrinsics: (B * N_cam, 3, 3) or (B, N_cam, 3, 3) depending on loader setup
            cam_extrinsics: (B * N_cam, 4, 4) or (B, N_cam, 4, 4)
            lidar_voxel_points: (N_voxels, max_points, 4)
            lidar_voxel_coords: (N_voxels, 4) -> [batch_idx, x, y, z]
        """
        B, N, C, H, W = camera_images.shape

        # --- CAMERA STREAM ---
        # Collapse batch and camera dimension for 2D Conv layers: [B, N, C, H, W] -> [B * N, C, H, W]
        camera_images_flat = camera_images.view(B * N, C, H, W)
        
        # Extract features per 2D camera view
        img_features = self.camera_backbone(camera_images_flat) # (B * N, camera_feat_dim, H_feat, W_feat)

        # Ensure geometric shapes are unrolled properly if they arrive packaged by batch
        if cam_intrinsics.dim() == 4:
            cam_intrinsics = cam_intrinsics.view(B * N, 3, 3)
        if cam_extrinsics.dim() == 4:
            cam_extrinsics = cam_extrinsics.view(B * N, 4, 4)

        # Lift 2D features into 3D world space and flatten to BEV grid
        camera_bev_features = self.view_transformer(img_features, cam_intrinsics, cam_extrinsics) # (B, combined_feat_dim, H_bev, W_bev)

        # --- LIDAR STREAM ---
        # Encode voxel point pillars into a matching 2D BEV matrix map
        lidar_bev_features = self.lidar_encoder(lidar_voxel_points, lidar_voxel_coords) # (B, combined_feat_dim, H_bev, W_bev)

        # Align batch structures if LiDAR stream defaults back down to singular constraints
        if lidar_bev_features.shape[0] < B:
            pad_size = B - lidar_bev_features.shape[0]
            padding = torch.zeros(pad_size, lidar_bev_features.shape[1], self.bev_h, self.bev_w, device=lidar_bev_features.device)
            lidar_bev_features = torch.cat([lidar_bev_features, padding], dim=0)

        # --- FUSION STREAM ---
        # Pack both independent spatial representations side by side
        fused_bev_features = torch.cat([camera_bev_features, lidar_bev_features], dim=1) # (B, combined_feat_dim * 2, H_bev, W_bev)
        fused_bev_features = self.fusion_conv(fused_bev_features) # (B, combined_feat_dim, H_bev, W_bev)

        # --- HEAD ---
        predictions = self.detection_head(fused_bev_features) # (B, 3, H_bev, W_bev)

        return predictions, fused_bev_features