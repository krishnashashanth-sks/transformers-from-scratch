import torch.optim as optim
from model import VisionTransformer,DINOHead
from torch.utils.data import DataLoader
from dataset import train_dataset,local_crops_number,cifar10_data_path
from train import train
from losses import DINOLoss
import torch
import os
import torchvision.transforms as transforms
from inference import get_dino_embedding
from PIL import Image
import numpy as np

# Define common architecture parameters for ViT and DINOHead
img_size = 224
patch_size = 16
in_chans = 3
embed_dim = 768  # Corresponds to ViT-B/16
depth = 12
num_heads = 12
mlp_ratio = 4.0
out_dim = 65536 # DINOv2 typically uses a large output dimension for the head
hidden_dim_list = [2048, 2048]

student_vit = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_chans,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio
)
teacher_vit = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_chans,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio
)

student_head = DINOHead(
    in_dim=embed_dim,
    out_dim=out_dim,
    hidden_dim_list=hidden_dim_list
)
teacher_head = DINOHead(
    in_dim=embed_dim,
    out_dim=out_dim,
    hidden_dim_list=hidden_dim_list
)

# Copy weights from student to teacher
teacher_vit.load_state_dict(student_vit.state_dict())
teacher_head.load_state_dict(student_head.state_dict())


# Step 4: Initialize torch.utils.data.DataLoader
batch_size = 32
num_workers = 2 # Changed from 4 to 2 to address UserWarning

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=False, # Set to False as no accelerator is found
    drop_last=True,  # Drop incomplete batches
    shuffle=True     # Shuffle dataset for each epoch
)

# Configuration for training
n_epochs = 100 # Total number of training epochs
warmup_teacher_temp = 0.04 # Initial teacher temperature
teacher_temp = 0.07 # Final teacher temperature
warmup_teacher_temp_epochs = 30 # Number of epochs for teacher temp warmup
student_temp = 0.1 # Student temperature
center_momentum = 0.9 # Center momentum for DINO loss

n_views = 2 + local_crops_number # Total number of crops (2 global + local_crops_number)
n_global_crops = 2 # Number of global crops

# 1. Initialize DINOLoss instance
dino_loss = DINOLoss(
    out_dim=out_dim,
    n_views=n_views,
    n_global_crops=n_global_crops,
    warmup_teacher_temp=warmup_teacher_temp,
    teacher_temp=teacher_temp,
    warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
    nepochs=n_epochs,
    student_temp=student_temp,
    center_momentum=center_momentum
)

# 2. Define an optimizer for the student network
# Using AdamW as it's common in ViT training
optimizer = optim.AdamW(student_vit.parameters(), lr=0.0005, weight_decay=0.04)

# 3. Set up the torch.device and move models/loss to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_vit.to(device)
teacher_vit.to(device)
student_head.to(device)
teacher_head.to(device)
dino_loss.to(device)

train(n_epochs,train_loader,student_vit,student_head,teacher_vit,teacher_head,dino_loss,optimizer,n_global_crops,batch_size,center_momentum,device)


inference_transform = transforms.Compose([
    transforms.Resize(img_size), # img_size is defined in previous cell as 224
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) )
])


print("Inference function 'get_dino_embedding' defined.")

# --- Usage Example ---
print("\n--- Demonstrating Inference Usage ---")

# Create a dummy image for demonstration purposes if needed
dummy_inference_image_path = './dummy_inference_image.png'
if not os.path.exists(dummy_inference_image_path):
    # Create a simple dummy image
    dummy_image_content = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(dummy_image_content).save(dummy_inference_image_path)
    print(f"Created a dummy inference image at {dummy_inference_image_path}")

try:
    # Get the DINOv2 embedding for the dummy image
    embedding = get_dino_embedding(dummy_inference_image_path,inference_transform,student_vit,student_head,device)
    print(f"Successfully obtained DINOv2 embedding.")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 embedding values: {embedding[0, :5].cpu().numpy()}")
except Exception as e:
    print(f"An error occurred during inference: {e}")
    print("Please ensure a valid image path is provided and the model was trained successfully.")
