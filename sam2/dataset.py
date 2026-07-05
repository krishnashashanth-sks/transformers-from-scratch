from torch.utils.data import Dataset
import torch
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, num_samples=100, img_size=256, embed_dim=768, num_point_labels=2, num_mask_tokens=4):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_point_labels = num_point_labels
        self.num_mask_tokens = num_mask_tokens

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.img_size, self.img_size)

        num_points = np.random.randint(0, 6) # Generate random number of points
        if num_points > 0:
            points = torch.rand(num_points, 2) * self.img_size
            point_labels = torch.randint(0, self.num_point_labels + 1, (num_points,))
        else:
            points = torch.empty(0, 2)
            point_labels = torch.empty(0, dtype=torch.long)

        num_boxes = np.random.randint(0, 3) # Generate random number of boxes
        if num_boxes > 0:
            boxes = torch.rand(num_boxes, 4) * self.img_size
        else:
            boxes = torch.empty(0, 4)

        ground_truth_mask = (torch.rand(1, self.img_size, self.img_size) > 0.5).float()
        original_image_size = torch.tensor([self.img_size, self.img_size])

        return {
            'image': image,
            'points': points,
            'point_labels': point_labels,
            'boxes': boxes,
            'ground_truth_mask': ground_truth_mask,
            'original_image_size': original_image_size
        }