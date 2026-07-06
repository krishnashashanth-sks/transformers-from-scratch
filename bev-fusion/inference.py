import torch

def run_inference(model, data_sample, device):
    model.eval()
    with torch.no_grad():
        # Prepare input data for inference (single sample or batch)
        camera_images = data_sample['camera_images'].unsqueeze(0).to(device) if data_sample['camera_images'].dim() == 3 else data_sample['camera_images'].to(device)
        cam_intrinsics = data_sample['cam_intrinsics'].unsqueeze(0).to(device) if data_sample['cam_intrinsics'].dim() == 2 else data_sample['cam_intrinsics'].to(device)
        cam_extrinsics = data_sample['cam_extrinsics'].unsqueeze(0).to(device) if data_sample['cam_extrinsics'].dim() == 2 else data_sample['cam_extrinsics'].to(device)
        lidar_voxel_points = data_sample['lidar_voxel_points'].to(device)
        # Adjust lidar_voxel_coords to batch_idx=0 for a single inference sample
        lidar_voxel_coords = data_sample['lidar_voxel_coords'].clone().to(device)
        lidar_voxel_coords[:, 0] = 0 # Ensure batch_idx is 0 for single inference

        predictions, fused_bev_features = model(
            camera_images, cam_intrinsics, cam_extrinsics,
            lidar_voxel_points, lidar_voxel_coords
        )
        return predictions, fused_bev_features