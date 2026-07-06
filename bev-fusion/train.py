
def train_one_epoch(num_epochs,model, dataloader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(dataloader):
        # Move data to device
        camera_images = data['camera_images'].to(device)
        cam_intrinsics = data['cam_intrinsics'].to(device)
        cam_extrinsics = data['cam_extrinsics'].to(device)
        lidar_voxel_points = data['lidar_voxel_points'].to(device)
        lidar_voxel_coords = data['lidar_voxel_coords'].to(device)
        
        # Targets are structured as a list of dicts from collate_fn. Extract first element for dummy target.
        # IMPORTANT: This assumes your `targets` dictionary inside `NuScenesDataset`'s `__getitem__` 
        # returns tensors that can be stacked and ultimately used by `criterion`.
        # For BEVFusion, targets are typically heatmaps, bounding box regressions, etc.
        # This needs to be correctly implemented in `NuScenesDataset` and matched with your actual `criterion`.
        target_heatmap = data['targets']['target_heatmap'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = model(
            camera_images, cam_intrinsics, cam_extrinsics,
            lidar_voxel_points, lidar_voxel_coords
        )
        
        # Calculate loss (predictions will be the output of detection_head)
        # Ensure predictions and targets have compatible shapes and types
        loss = criterion(predictions, target_heatmap) # Use target_heatmap as dummy target for MSE
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_loss:.4f}")
    return avg_loss
