from transform import MultiCropTransform
import torchvision.datasets as datasets
import os

global_crops_scale = (0.4, 1.0)
local_crops_scale = (0.05, 0.4)
local_crops_number = 8  # As per DINOv2, typically 2 global + 8 local views

multi_crop_transform = MultiCropTransform(
    global_crops_scale=global_crops_scale,
    local_crops_scale=local_crops_scale,
    local_crops_number=local_crops_number,
    global_crop_size=224, # DINOv2 typically uses 224 for global crops
    local_crop_size=96    # DINOv2 typically uses 96 for local crops
)

class DINOv2CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root, train, transform, download=True):
        super().__init__(root, train=train, download=download, transform=None)
        self.dino_transform = transform # This will be our MulitCropTransform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # The transform (MulitCropTransform) will return a list of augmented views
        multi_crop_views = self.dino_transform(img)
        return multi_crop_views # DINO typically doesn't use labels directly in the SSL phase
    
cifar10_data_path = './data/cifar10'
os.makedirs(cifar10_data_path, exist_ok=True)

train_dataset = DINOv2CIFAR10Dataset(
    root=cifar10_data_path,
    train=True,
    download=True,
    transform=multi_crop_transform
)