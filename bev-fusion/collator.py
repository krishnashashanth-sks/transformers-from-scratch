import torch

def collate_fn(batch):
    # This function needs to handle batching of variable-sized tensors (like voxel_points and voxel_coords).
    # For tensors with fixed dimensions (images, intrinsics, extrinsics), use torch.stack.
    # For variable dimensions, you might need to concatenate and keep track of batch indices.

    batched_data = {}
    
    # Example: Fixed-size tensors can be stacked directly
    batched_data['camera_images'] = torch.stack([item['camera_images'] for item in batch]) # (B*N_cam, C, H, W)
    batched_data['cam_intrinsics'] = torch.stack([item['cam_intrinsics'] for item in batch]) # (B*N_cam, 3, 3)
    batched_data['cam_extrinsics'] = torch.stack([item['cam_extrinsics'] for item in batch]) # (B*N_cam, 4, 4)

    # Handle variable-sized LiDAR voxel data
    # We need to add a batch_idx to lidar_voxel_coords and concatenate
    lidar_voxel_points_batch = []
    lidar_voxel_coords_batch = []
    for i, item in enumerate(batch):
        # Adjust batch index for each sample's voxel_coords
        # Only append if there are actual voxels for this sample
        if item['lidar_voxel_coords'].numel() > 0:
            coords = item['lidar_voxel_coords'].clone() # (N_voxels, 4)
            coords[:, 0] = i # Set batch index
            
            lidar_voxel_points_batch.append(item['lidar_voxel_points'])
            lidar_voxel_coords_batch.append(coords)
    
    if len(lidar_voxel_points_batch) > 0:
        batched_data['lidar_voxel_points'] = torch.cat(lidar_voxel_points_batch, dim=0)
        batched_data['lidar_voxel_coords'] = torch.cat(lidar_voxel_coords_batch, dim=0)
    else:
        # If no voxels across the entire batch, return empty tensors of correct shape
        # Assume a dummy shape for inner dimensions if no voxels are present in any batch item
        batched_data['lidar_voxel_points'] = torch.empty(0, batch[0]['lidar_voxel_points'].shape[1], batch[0]['lidar_voxel_points'].shape[2])
        batched_data['lidar_voxel_coords'] = torch.empty(0, 4).long()

    # Handle targets - might also need custom collate logic depending on their structure
    # For dummy targets, we can try to stack or create lists
    # This assumes all items in batch have the same target structure with Tensors
    if 'targets' in batch[0] and isinstance(batch[0]['targets'], dict):
        batched_data['targets'] = {}
        for key in batch[0]['targets']:
            if isinstance(batch[0]['targets'][key], torch.Tensor):
                batched_data['targets'][key] = torch.stack([b['targets'][key] for b in batch])
            else:
                batched_data['targets'][key] = [b['targets'][key] for b in batch] # Fallback for non-tensor targets
    else: # If targets are not a dict or not present
        batched_data['targets'] = [item['targets'] for item in batch]


    batched_data['sample_tokens'] = [item['sample_token'] for item in batch]

    return batched_data
