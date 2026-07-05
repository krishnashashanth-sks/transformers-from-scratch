import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from model import VisionTransformer
from train import train_epoch
from evaluate import validate_epoch
from inference import predict_tensor_image
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import time

# Define image transformations for training and validation
# Assuming img_size = 224 based on ViT architecture details
img_size = 224

transform_train = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 mean/std
])

transform_val = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
# Using a small dataset for demonstration purposes due to 'from scratch' training
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

# Create data loaders
batch_size =16# A common batch size for training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"CIFAR-10 training dataset loaded with {len(train_dataset)} samples.")
print(f"CIFAR-10 validation dataset loaded with {len(val_dataset)} samples.")
print(f"Train loader created with batch size {batch_size}.")
print(f"Validation loader created with batch size {batch_size}.")

# Define loss function and optimizer
# For classification tasks, CrossEntropyLoss is suitable.
# AdamW is a common choice for Transformers due to its weight decay regularization.

# Instantiate the Vision Transformer model (ensure n_classes matches CIFAR-10, which is 10)
model = VisionTransformer(img_size=img_size, n_classes=10) # CIFAR-10 has 10 classes

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"Model initialized and moved to: {device}")
# Training loop parameters
num_epochs = 10 # Starting with a small number of epochs for demonstration

print("Starting training...")

for epoch in tqdm(range(num_epochs)):
    start_time = time.time()

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')

print("Training finished.")
# Get class names from CIFAR-10 dataset
class_names = val_dataset.classes

# Select an image from the validation set for demonstration
# Let's take the first image from the validation dataset
example_image, example_label_idx = val_dataset[0]

predicted_class, predicted_prob = predict_tensor_image(model, example_image, class_names, device)

print(f"True label: {class_names[example_label_idx]}")
print(f"Predicted class: {predicted_class}")
print(f"Prediction probability: {predicted_prob:.4f}")