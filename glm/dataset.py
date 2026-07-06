from torch.utils.data import Dataset
import torch
from utils import preprocess_text

class GLM5Dataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        input_ids, attention_mask = preprocess_text(text, self.tokenizer, self.max_len)

        # For training a language model, the labels are typically the next tokens in the sequence.
        # In a simple generative setup like this, the input sequence shifted by one is often used as labels.
        # The target for each token is the token immediately following it in the sequence.
        # If the input is [t1, t2, t3], the labels are [t2, t3, <PAD_ID>].
        # We need to make sure labels are also padded/truncated to max_len.
        labels = torch.cat((input_ids[1:], torch.tensor([self.tokenizer.pad_id])))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        