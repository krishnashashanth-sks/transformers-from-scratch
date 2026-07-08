from torch.utils.data import Dataset
from torchvision import transforms
from tokenizer import BBoxTokenizer
import torch

# --- Conceptual MultimodalDataset  ---
class MultimodalDataset(Dataset):
    def __init__(self, data_list: list, image_size: int, max_seq_len: int,
                 bbox_tokenizer: BBoxTokenizer, text_tokenizer_fn, padding_token_id: int = 0):
        self.data_list = data_list
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.bbox_tokenizer = bbox_tokenizer
        self.text_tokenizer_fn = text_tokenizer_fn
        self.padding_token_id = padding_token_id

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text_raw = item['text']
        bbox_annotations = item.get('bbox_annotations', [])

        image = torch.randn(3, self.image_size, self.image_size) # dummy image

        token_ids = []
        word_tokens = self.text_tokenizer_fn(text_raw) # Dummy word tokens
        token_ids.extend(word_tokens)

        for ann in bbox_annotations:
            bbox_coords = ann['bbox']
            bbox_token_sequence = self.bbox_tokenizer.tokenize_bbox(bbox_coords)
            token_ids.extend(bbox_token_sequence)

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids = token_ids + [self.padding_token_id] * (self.max_seq_len - len(token_ids))

        text_tokens = torch.tensor(token_ids, dtype=torch.long)

        return image, text_tokens

