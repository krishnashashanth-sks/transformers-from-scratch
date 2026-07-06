import torch
from torch.utils.data import Dataset
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import os
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter 
from pyquaternion import Quaternion 

class NuScenesDataset(Dataset):
    def __init__(self, nusc, version='v1.0-mini', dataroot='/content/drive/MyDrive/nuscenes',
                 image_size=(224, 224), bev_range=([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]),
                 voxel_size=(0.1, 0.1, 0.2), max_points_per_voxel=32, num_cameras=6,
                 bev_h=200, bev_w=200, split='train'): 
        self.nusc = nusc
        self.version = version
        self.dataroot = dataroot
        self.image_size = image_size 
        self.bev_range = bev_range 
        self.max_points_per_voxel = max_points_per_voxel
        self.num_cameras = num_cameras

        self.bev_h = bev_h
        self.bev_w = bev_w

        # Dynamically calculate voxel_size based on bev_range and desired bev_h, bev_w
        x_min, y_min, z_min, x_max, y_max, z_max = self.bev_range
        calculated_voxel_size_x = (x_max - x_min) / self.bev_w
        calculated_voxel_size_y = (y_max - y_min) / self.bev_h
        self.voxel_size = (calculated_voxel_size_x, calculated_voxel_size_y, voxel_size[2]) 
        print(f"Calculated voxel_size: {self.voxel_size}")

        # Define image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        # Filter samples based on the split (e.g., 'train', 'val', 'mini_train', 'mini_val')
        self.sample_tokens = self._get_sample_tokens_for_split(split)

        # Prepare camera sensor tokens
        self.camera_channels = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
        ]
        assert len(self.camera_channels) == self.num_cameras, \
            f"Expected {self.num_cameras} camera channels, but got {len(self.camera_channels)}"

        print(f"Initialized NuScenesDataset with {len(self.sample_tokens)} samples for split '{split}'.")

    def _get_sample_tokens_for_split(self, split):
        if split == 'mini_train':
            scene_names = [
                'scene-0061', 'scene-0065', 'scene-0103', 'scene-0106',
                'scene-0107', 'scene-0109', 'scene-0150', 'scene-0151'
            ]
        elif split == 'mini_val':
            scene_names = [
                'scene-0152', 'scene-0153', 'scene-0154', 'scene-0168'
            ] 
        else: 
            scene_names = [s['name'] for s in self.nusc.scene]

        tokens = []
        for sample in self.nusc.sample:
            scene_record = self.nusc.get('scene', sample['scene_token'])
            if scene_record['name'] in scene_names:
                tokens.append(sample['token'])
        return tokens

    def __len__(self):
        return len(self.sample_tokens)

    def _load_camera_data(self, sample_record):
        camera_images = []
        camera_intrinsics = []
        camera_extrinsics = []

        for channel in self.camera_channels:
            sd_record = self.nusc.get('sample_data', sample_record['data'][channel])

            # Load image
            image_path = os.path.join(self.nusc.dataroot, sd_record['filename'])
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            camera_images.append(image)

            # Get camera intrinsic matrix
            calibrated_sensor_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            intrinsic = np.array(calibrated_sensor_record['camera_intrinsic'])
            camera_intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))

            # Get sensor-to-ego pose (extrinsics)
            sensor_to_ego = transform_matrix(
                calibrated_sensor_record['translation'],
                Quaternion(calibrated_sensor_record['rotation']), 
                inverse=False
            )
            # Get ego-to-global pose
            ego_pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            ego_to_global = transform_matrix(
                ego_pose_record['translation'],
                Quaternion(ego_pose_record['rotation']), 
                inverse=False
            )

            camera_extrinsics.append(torch.tensor(ego_to_global @ sensor_to_ego, dtype=torch.float32))

        return torch.stack(camera_images), torch.stack(camera_intrinsics), torch.stack(camera_extrinsics)

    def _load_lidar_data(self, sample_record):
        lidar_sd_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.nusc.dataroot, lidar_sd_record['filename'])

        # Load point cloud
        points = LidarPointCloud.from_file(lidar_path)

        # Get lidar-to-ego pose (Fixed with Quaternion wrapper)
        calibrated_sensor_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        lidar_to_ego = transform_matrix(
            calibrated_sensor_record['translation'],
            Quaternion(calibrated_sensor_record['rotation']), 
            inverse=False
        )

        # Get ego-to-global pose (Fixed with Quaternion wrapper)
        ego_pose_record = self.nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
        ego_to_global = transform_matrix(
            ego_pose_record['translation'],
            Quaternion(ego_pose_record['rotation']), 
            inverse=False
        )

        # Transform points from lidar frame to global frame
        global_from_lidar = ego_to_global @ lidar_to_ego
        points.transform(global_from_lidar)

        # Get points as numpy array: (4, N) for (x, y, z, intensity)
        points_np = points.points.T 

        # Filter points within BEV range
        x_min, y_min, z_min, x_max, y_max, z_max = self.bev_range
        mask_x = (points_np[:, 0] >= x_min) & (points_np[:, 0] < x_max)
        mask_y = (points_np[:, 1] >= y_min) & (points_np[:, 1] < y_max)
        mask_z = (points_np[:, 2] >= z_min) & (points_np[:, 2] < z_max)
        valid_points = points_np[mask_x & mask_y & mask_z]

        if len(valid_points) == 0:
            return torch.empty(0, self.max_points_per_voxel, 4), torch.empty(0, 4).long()

        # Convert global coordinates to voxel coordinates
        x_coords = ((valid_points[:, 0] - x_min) / self.voxel_size[0]).astype(int)
        y_coords = ((valid_points[:, 1] - y_min) / self.voxel_size[1]).astype(int)
        z_coords = ((valid_points[:, 2] - z_min) / self.voxel_size[2]).astype(int)

        # Stack coordinates to form (N_points, 3) voxel indices
        voxel_indices = np.stack([x_coords, y_coords, z_coords], axis=1)

        grid_x_dim = int((x_max - x_min) / self.voxel_size[0])
        grid_y_dim = int((y_max - y_min) / self.voxel_size[1])
        grid_z_dim = int((z_max - z_min) / self.voxel_size[2])

        # Ensure voxel indices are within bounds
        voxel_indices[:, 0] = np.clip(voxel_indices[:, 0], 0, grid_x_dim - 1)
        voxel_indices[:, 1] = np.clip(voxel_indices[:, 1], 0, grid_y_dim - 1)
        voxel_indices[:, 2] = np.clip(voxel_indices[:, 2], 0, grid_z_dim - 1)

        unique_voxels, inverse_indices, counts = np.unique(
            voxel_indices, axis=0, return_inverse=True, return_counts=True
        )

        num_unique_voxels = len(unique_voxels)

        # Initialize tensors for voxel points and coordinates
        voxel_points_tensor = torch.zeros(
            num_unique_voxels, self.max_points_per_voxel, 4, dtype=torch.float32
        )
        voxel_coords_tensor = torch.zeros(
            num_unique_voxels, 4, dtype=torch.long
        )

        # Store unique voxel coordinates (batch idx is index 0)
        voxel_coords_tensor[:, 1:] = torch.from_numpy(unique_voxels).long()

        # Fill voxel_points_tensor
        points_per_voxel_counter = np.zeros(num_unique_voxels, dtype=int)
        for i, point in enumerate(valid_points):
            voxel_idx = inverse_indices[i] 
            if points_per_voxel_counter[voxel_idx] < self.max_points_per_voxel:
                voxel_points_tensor[voxel_idx, points_per_voxel_counter[voxel_idx]] = torch.from_numpy(point)
                points_per_voxel_counter[voxel_idx] += 1

        return voxel_points_tensor, voxel_coords_tensor

    def _load_annotations_and_generate_targets(self, sample_record):
        center_heatmap_raw = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)
        x_min, y_min, _, x_max, y_max, _ = self.bev_range

        for ann_token in sample_record['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            center_x, center_y, _ = ann_record['translation']

            # Convert global coordinates to BEV pixel coordinates
            px = int(((center_x - x_min) / (x_max - x_min)) * self.bev_w)
            py = int(((center_y - y_min) / (y_max - y_min)) * self.bev_h)

            px = np.clip(px, 0, self.bev_w - 1)
            py = np.clip(py, 0, self.bev_h - 1)

            center_heatmap_raw[py, px] = 1.0 

        # Apply Gaussian blur to the center heatmap
        smoothed_center_heatmap = gaussian_filter(center_heatmap_raw, sigma=2)

        if smoothed_center_heatmap.max() > 0:
            smoothed_center_heatmap = smoothed_center_heatmap / smoothed_center_heatmap.max()

        target_heatmap = np.stack([smoothed_center_heatmap, smoothed_center_heatmap, smoothed_center_heatmap], axis=0)

        return {
            'target_heatmap': torch.from_numpy(target_heatmap).float()
        }

    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        sample_record = self.nusc.get('sample', sample_token)

        camera_images, camera_intrinsics, camera_extrinsics = self._load_camera_data(sample_record)
        lidar_voxel_points, lidar_voxel_coords = self._load_lidar_data(sample_record)
        targets = self._load_annotations_and_generate_targets(sample_record)

        return {
            'camera_images': camera_images, 
            'cam_intrinsics': camera_intrinsics, 
            'cam_extrinsics': camera_extrinsics, 
            'lidar_voxel_points': lidar_voxel_points, 
            'lidar_voxel_coords': lidar_voxel_coords, 
            'targets': targets,
            'sample_token': sample_token
        }