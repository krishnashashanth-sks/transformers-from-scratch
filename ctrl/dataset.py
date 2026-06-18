from torch.utils.data import Dataset, DataLoader
import torch

class DummyTextDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size, control_code_range=(0, 5)): # Control codes 0-4
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.control_code_range = control_code_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate a control code at the beginning (e.g., a single token)
        control_code = torch.randint(self.control_code_range[0], self.control_code_range[1], (1,))

        # Simulate text tokens, ensuring they don't overlap with small control code IDs
        text_tokens = torch.randint(self.control_code_range[1], self.vocab_size, (self.seq_len - 1,))

        # Concatenate control code and text tokens
        input_ids = torch.cat((control_code, text_tokens))

        # Labels are the input_ids shifted. The model predicts input_ids[i+1] from input_ids[i].
        # The first label is typically ignored or masked, as there's no preceding token to predict it.
        labels = torch.cat((text_tokens, torch.tensor([self.vocab_size - 1])))
        # For simplicity, we'll just shift it. A real LM would often predict the next token
        # for all positions up to (seq_len-1), and the label at position i is input_ids[i+1].
        # The last label is usually the EOS token or a special padding token.

        # In a real setup, labels would be input_ids[:, 1:] and inputs would be input_ids[:, :-1]
        # For our current model, we'll use input_ids as both input and target, and manually shift for loss

        # Create a dummy attention mask (all ones for now, assuming no padding)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)

        return {"input_ids": input_ids.long(), "labels": input_ids.long(), "attention_mask": attention_mask.long()}

# Instantiate a dummy dataset and DataLoader
dummy_dataset = DummyTextDataset(num_samples=100, seq_len=50, vocab_size=VOCAB_SIZE)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)