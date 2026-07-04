from torch.utils.data import Dataset
import torch

# Custom Dataset class to wrap the tokenized_datasets
class LanguageModelingDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokenized_dataset[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.tokenized_dataset[idx]['attention_mask'], dtype=torch.long)
        }