from torch.utils.data import Dataset

class CustomByteDataset(Dataset):
    def __init__(self, preprocessed_sequences, pad_token):
        self.data = preprocessed_sequences
        self.pad_token = pad_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_sequence = self.data[idx]

        encoder_input = full_sequence
        decoder_input = full_sequence[:-1]
        decoder_target = full_sequence[1:]

        return encoder_input, decoder_input, decoder_target
