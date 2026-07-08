import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import DualAttentionVisionTransformer
from train import train_model
import torch.nn as nn
import torch

img_size = 224
patch_size = 16
in_channels = 3
num_classes = 10
batch_size = 4
sequence_length = (img_size // patch_size) ** 2 + 1 # num_patches + cls_token
embed_dim = 768


# Define transformations for CIFAR-100
# Resize to img_size (224x224), convert to tensor, and normalize
transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Download and load the CIFAR-100 training and test datasets
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Update num_classes for CIFAR-100
num_classes_cifar100 = 100

print(f"CIFAR-100 training dataset created with {len(train_dataset)} samples.")
print(f"CIFAR-100 test dataset created with {len(test_dataset)} samples.")
print(f"Updated number of classes for CIFAR-100: {num_classes_cifar100}")
# --- Model, Loss, and Optimizer Initialization ---

# Instantiate the model
model = DualAttentionVisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes_cifar100, # Updated to CIFAR-100's number of classes
    embed_dim=embed_dim,
    depth=2, # Reduced depth for quicker demo
    num_heads=8,
    mlp_ratio=4.,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    group_size=7 # Ensure img_size // patch_size is divisible by group_size
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Loss Function
criterion = nn.CrossEntropyLoss()

# Define Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

print(f"Model instantiated and moved to {device}.")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

num_epochs = 3 # Reduced epochs for quicker demo

train_model(num_epochs,model,train_dataloader,test_dataloader,optimizer,criterion,device)

# Get class names for CIFAR-100
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Set the model to evaluation mode
model.eval()

# Get 5 samples from the test_dataloader
samples_to_visualize = 5
fig, axes = plt.subplots(1, samples_to_visualize, figsize=(15, 5))

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_dataloader):
        if i >= samples_to_visualize:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Unnormalize image for display
        # CIFAR-100 normalization was (0.5, 0.5, 0.5) for mean and std
        img = inputs[0].cpu().numpy()
        img = img / 2 + 0.5  # unnormalize
        img = np.transpose(img, (1, 2, 0))

        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"True: {cifar100_classes[labels[0].item()]}\nPred: {cifar100_classes[predicted[0].item()]}")
        ax.axis('off')

plt.tight_layout()
plt.show()