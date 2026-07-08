from tokenizer import DummyTokenizer
import torchvision.transforms as transforms
import os
from dataset import MultimodalDataset
from torch.utils.data import DataLoader
from PIL import Image

def create_dummy_image(path, size=(64, 64), color=(255, 0, 0)):
    img = Image.new('RGB', size, color)
    img.save(path)

def setup_data(dummy_image_dir, image_paths, dummy_data, MAX_SEQ_LEN, NUM_VISUAL_TOKENS, BATCH_SIZE):
    os.makedirs(dummy_image_dir, exist_ok=True)

    for i, path in enumerate(image_paths):
        create_dummy_image(path, color=(i*50, 255 - i*50, 100))

    dummy_tokenizer = DummyTokenizer()
    dummy_image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    multimodal_dataset = MultimodalDataset(
        data=dummy_data,
        tokenizer=dummy_tokenizer,
        image_transform=dummy_image_transform,
        max_seq_len=MAX_SEQ_LEN,
        num_visual_tokens=NUM_VISUAL_TOKENS
    )

    multimodal_dataloader = DataLoader(
        multimodal_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return dummy_tokenizer, dummy_image_transform, multimodal_dataset, multimodal_dataloader

