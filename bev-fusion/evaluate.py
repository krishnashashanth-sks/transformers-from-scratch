import torch

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            camera_images = data['camera_images'].to(device)
            cam_intrinsics = data['cam_intrinsics'].to(device)
            cam_extrinsics = data['cam_extrinsics'].to(device)
            lidar_voxel_points = data['lidar_voxel_points'].to(device)
            lidar_voxel_coords = data['lidar_voxel_coords'].to(device)
            target_heatmap = data['targets'][0]['target_heatmap'].to(device) # Example: Accessing dummy target
            
            predictions, _ = model(
                camera_images, cam_intrinsics, cam_extrinsics,
                lidar_voxel_points, lidar_voxel_coords
            )
            loss = criterion(predictions, target_heatmap)
            total_loss += loss.item()
            
            # Example of collecting predictions for metric calculation (needs actual implementation)
            # all_predictions.append(predictions.cpu().numpy())
            # all_targets.append(data['targets'][0]['actual_target'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation finished. Average Validation Loss: {avg_loss:.4f}")
    return avg_loss