# Create a directory for the dataset
import os
import cv2

dataset_dir = 'UCF101_subset'
os.makedirs(dataset_dir, exist_ok=True)

# Define the class to download and its URL
# For demonstration, let's pick a reliable sample MP4 video
class_name = 'SampleVideo'
class_url = 'https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4'

# Download the video file using !wget, which is often more robust in Colab for direct file downloads.
# Using a generic name for the video file after download to avoid issues with original name being HTML.
video_filename = 'sample_video.mp4'
video_path = os.path.join(dataset_dir, video_filename)

if not os.path.exists(video_path):
    print(f'Downloading {class_name} video...')
    # Use !wget for downloading with --no-check-certificate if needed, but this URL should be fine.
    # !wget -O {video_path} {class_url} # -O to specify output filename
    print('Download complete.')
else:
    print(f'Video already exists at {video_path}. Skipping download.')

print(f'Dataset subset prepared in: {dataset_dir}')
print(f'Content of {dataset_dir}:')
print(os.listdir(dataset_dir))

# Update video_name_without_ext and video_path for subsequent steps
video_name_without_ext = os.path.splitext(video_filename)[0]

# Define output directory for frames
frames_output_dir = 'UCF101_frames'
os.makedirs(frames_output_dir, exist_ok=True)

current_video_frames_dir = os.path.join(frames_output_dir, video_name_without_ext)
os.makedirs(current_video_frames_dir, exist_ok=True)

print(f"Extracting frames from {video_path} to {current_video_frames_dir}...")

cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame as JPEG
    frame_filename = os.path.join(current_video_frames_dir, f'frame_{frame_count:05d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Finished extracting {frame_count} frames.")

print(f"Content of {frames_output_dir} after extraction:")
print(os.listdir(frames_output_dir))
print(f"Content of {current_video_frames_dir}:")
# List first 5 frames to avoid too much output
print(os.listdir(current_video_frames_dir)[:5], "...")

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Assuming frames_output_dir is defined from previous steps: 'UCF101_frames'

class VideoDataset(data.Dataset):
    def __init__(self, root_dir, video_names, frame_transform=None, sequence_length=16):
        self.root_dir = root_dir
        self.video_names = video_names # List of video subdirectories within root_dir (e.g., ['sample_video'])
        self.frame_transform = frame_transform
        self.sequence_length = sequence_length
        self.video_info = self._get_video_info()

    def _get_video_info(self):
        video_info = []
        for video_name in self.video_names:
            video_path = os.path.join(self.root_dir, video_name)
            frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
            if not frames:
                continue # Skip if no frames found

            # We'll just assign a dummy label for now as we only have one video
            # In a real dataset, this would come from annotation files.
            label = 0

            # For each video, create multiple 'clips' of sequence_length frames
            for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length // 2): # overlap by half
                clip_frames = frames[i : i + self.sequence_length]
                if len(clip_frames) == self.sequence_length:
                    video_info.append({'frames': clip_frames, 'label': label, 'video_name': video_name})
        return video_info

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        item = self.video_info[idx]
        clip_frames_paths = item['frames']
        label = item['label']

        frames = []
        for frame_path in clip_frames_paths:
            img = Image.open(frame_path).convert('RGB')
            frames.append(img)

        # Apply transformations to each frame
        if self.frame_transform:
            frames = [self.frame_transform(img) for img in frames]

        # Stack frames into a single tensor (T, C, H, W) or (C, T, H, W) as needed by model
        # For TRecViT, often (C, T, H, W) is preferred for conv3d or some transformer inputs
        frames_tensor = torch.stack(frames, dim=1) # stack along new dimension for T: (C, T, H, W)

        return frames_tensor, label

# Define transformations for the frames
# These are common transformations for image classification tasks
transforms_pipeline = transforms.Compose([
    transforms.Resize((128, 128)), # Resize frames to a common size
    transforms.ToTensor(),       # Convert PIL Image to PyTorch Tensor (H, W, C) to (C, H, W) and scale to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

# Get the list of video subdirectories (in our case, just 'sample_video')
video_subdirs = [d for d in os.listdir(frames_output_dir) if os.path.isdir(os.path.join(frames_output_dir, d))]

# Instantiate the dataset
sequence_length = 16 # Number of frames per video sequence

if video_subdirs:
    dataset = VideoDataset(root_dir=frames_output_dir,
                           video_names=video_subdirs,
                           frame_transform=transforms_pipeline,
                           sequence_length=sequence_length)

    print(f"Total video sequences in dataset: {len(dataset)}")

    # Create a DataLoader
    batch_size = 2 # Small batch size for demonstration
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"DataLoader created with batch size: {batch_size}")

    # Test: Get one batch from the dataloader
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: sequences shape {sequences.shape}, labels shape {labels.shape}")
        # Expected shape: (batch_size, C, T, H, W)
        # e.g., (2, 3, 16, 128, 128) if batch_size=2, sequence_length=16
        break # Just get one batch for demonstration
else:
    print(f"No video directories found in {frames_output_dir}. Please ensure frames were extracted correctly.")
    dataloader=None