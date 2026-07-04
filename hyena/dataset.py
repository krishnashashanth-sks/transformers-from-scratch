from torch.utils.data import Dataset
import torch

class LanguageModelingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data["input_ids"][idx], dtype=torch.long),
            "labels": torch.tensor(self.data["labels"][idx], dtype=torch.long),
        }