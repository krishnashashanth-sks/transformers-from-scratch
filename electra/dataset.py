from torch.utils.data import Dataset
import torch

# Define a custom dataset for tokenization
class WikitextDataset(Dataset):
    def __init__(self, encodings, max_seq_len, pad_token_id):
        self.encodings = encodings
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Directly get the dictionary for the sample at idx from the datasets.Dataset object
        sample = self.encodings[idx]
        item = {key: torch.tensor(val, dtype=torch.long) for key, val in sample.items()}

        # Ensure all tensors are MAX_SEQ_LEN and pad if necessary
        # Truncate if longer than max_seq_len
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            if item[key].size(0) > self.max_seq_len:
                item[key] = item[key][:self.max_seq_len]
            elif item[key].size(0) < self.max_seq_len:
                padding_length = self.max_seq_len - item[key].size(0)
                pad_value = self.pad_token_id if key == 'input_ids' else 0
                item[key] = torch.cat([item[key], torch.full((padding_length,), pad_value, dtype=torch.long)])

        # original_labels are simply the input_ids before any masking for MLM target.
        # This will be masked by ElectraModel._apply_masking_and_replacement internally.
        item['original_labels'] = item['input_ids'].clone()
        return item
