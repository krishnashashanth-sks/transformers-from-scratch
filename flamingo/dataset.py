from torch.utils.data import Dataset
from PIL import Image
import torch

class MultimodalDataset(Dataset):
    def __init__(
        self,
        data: list,
        tokenizer,
        image_transform,
        max_seq_len: int,
        num_visual_tokens: int
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len
        self.num_visual_tokens = num_visual_tokens
        self.IMG_TOKEN_ID = self.tokenizer.get_img_token_id()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text_caption = self.data[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        text_token_ids = self.tokenizer.tokenize(text_caption)

        img_placeholders = [self.IMG_TOKEN_ID] * self.num_visual_tokens

        max_text_len = self.max_seq_len - self.num_visual_tokens
        if max_text_len < 0:
            raise ValueError("max_seq_len must be greater than or equal to num_visual_tokens")

        processed_text_token_ids = self.tokenizer.pad(text_token_ids, max_text_len)

        final_lm_input_token_ids = img_placeholders + processed_text_token_ids

        assert len(final_lm_input_token_ids) == self.max_seq_len,f"Final sequence length mismatch: {len(final_lm_input_token_ids)} != {self.max_seq_len}"

        return image, torch.tensor(final_lm_input_token_ids, dtype=torch.long)

