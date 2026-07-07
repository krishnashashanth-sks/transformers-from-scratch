import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(self, encoded_texts, max_seq_len, pad_token_id, eos_token_id):
        self.encoded_texts = encoded_texts
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        # Get token IDs from the tokenizer's Encoding object
        ids = self.encoded_texts[idx].ids

        # The `BertProcessing` post-processor adds `<s>` and `</s>` tokens.
        # For next-token prediction, we use `<s> T1 T2 ... Tn` as input and predict `T1 T2 ... Tn </s>`.
        # So, input_ids will be `ids[:-1]` and target_ids will be `ids[1:]`.
        input_ids = ids[:-1]  # Exclude the final </s> token from input
        target_ids = ids[1:]   # Exclude the initial <s> token from target

        # Truncate sequences if they are longer than max_seq_len
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            target_ids = target_ids[:self.max_seq_len] # Target must match input length after truncation

        # Pad sequences to max_seq_len
        padding_length = self.max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids_padded = input_ids + [self.pad_token_id] * padding_length
            target_ids_padded = target_ids + [self.pad_token_id] * padding_length
        else: # No padding needed or truncated to exact max_seq_len
            input_ids_padded = input_ids
            target_ids_padded = target_ids

        input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long)
        target_ids_tensor = torch.tensor(target_ids_padded, dtype=torch.long)

        return input_ids_tensor, target_ids_tensor
