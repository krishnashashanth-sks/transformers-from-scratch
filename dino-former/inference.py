from PIL import Image
import torch

def get_dino_embedding(image_path,inference_transform,student_vit,student_head,device):
    """
    Loads an image, preprocesses it, and returns its DINOv2 embedding.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: The DINOv2 embedding of the image.
    """
    # 1. Load the image
    img = Image.open(image_path).convert("RGB") # Ensure RGB, even for grayscale, then let transform handle channels
    
    
    # 2. Preprocess the image
    preprocessed_image = inference_transform(img)
    # Add a batch dimension (B, C, H, W)
    preprocessed_image = preprocessed_image.unsqueeze(0).to(device)

    # 3. Get embedding using the student model
    with torch.no_grad():
        vit_output_embedding = student_vit(preprocessed_image)
        dino_embedding = student_head(vit_output_embedding)

    return dino_embedding
