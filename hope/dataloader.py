import torch

# Simulate a DataLoader for a validation set
class SimulatedDataLoader:
    def __init__(self, vocab_size, batch_size, seq_len, num_batches):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), dtype=torch.long)
            target_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), dtype=torch.long)
            yield input_ids, target_ids

    def __len__(self):
        return self.num_batches