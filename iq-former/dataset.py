import torch
from utils import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

synthetic_data = [
    ("The cat sat on the mat.", "Le chat s'est assis sur le tapis."),
    ("Dog is barking loudly.", "Le chien aboie bruyamment."),
    ("I like to eat apples.", "J'aime manger des pommes."),
    ("She is reading a book.", "Elle lit un livre."),
    ("We are going to the park.", "Nous allons au parc.")
]

class Vocab:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.n_words = 4  # Count special tokens

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def add_sentence(self, sentence):
        for word in tokenize_sentence(sentence):
            self.add_word(word)

# Initialize source and target vocabularies
src_vocab = Vocab()
tgt_vocab = Vocab()

# Populate vocabularies from synthetic data
for src_sentence, tgt_sentence in synthetic_data:
    src_vocab.add_sentence(src_sentence)
    tgt_vocab.add_sentence(tgt_sentence)

prepared_data = []

sos_id = tgt_vocab.word2idx['<sos>']
eos_id = tgt_vocab.word2idx['<eos>']

for src_sentence, tgt_sentence in synthetic_data:
    # Tokenize and numericalize source
    src_tokens = tokenize_sentence(src_sentence)
    src_ids = tokens_to_ids(src_tokens, src_vocab)

    # Tokenize and numericalize target, and add <sos>/<eos>
    tgt_tokens = tokenize_sentence(tgt_sentence)
    tgt_input_ids = [sos_id] + tokens_to_ids(tgt_tokens, tgt_vocab) # Input to decoder
    tgt_output_ids = tokens_to_ids(tgt_tokens, tgt_vocab) + [eos_id] # Ground truth for loss calculation

    prepared_data.append({
        'src_ids': src_ids,
        'tgt_input_ids': tgt_input_ids,
        'tgt_output_ids': tgt_output_ids,
        'src_text': src_sentence,
        'tgt_text': tgt_sentence
    })


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert lists to torch tensors
        src_ids = torch.tensor(sample['src_ids'], dtype=torch.long)
        tgt_input_ids = torch.tensor(sample['tgt_input_ids'], dtype=torch.long)
        tgt_output_ids = torch.tensor(sample['tgt_output_ids'], dtype=torch.long)

        return {
            'src_ids': src_ids,
            'tgt_input_ids': tgt_input_ids,
            'tgt_output_ids': tgt_output_ids
        }

# Instantiate the dataset
translation_dataset = TranslationDataset(prepared_data)

def collate_fn(batch):
    # Extract sequences from the batch
    src_ids_batch = [item['src_ids'] for item in batch]
    tgt_input_ids_batch = [item['tgt_input_ids'] for item in batch]
    tgt_output_ids_batch = [item['tgt_output_ids'] for item in batch]

    # Get padding indices
    src_pad_idx = src_vocab.word2idx['<pad>']
    tgt_pad_idx = tgt_vocab.word2idx['<pad>']

    # Pad sequences to the maximum length in the batch
    # pad_sequence pads to the length of the longest sequence in the list.
    # batch_first=True makes the output tensor (batch_size, seq_len)
    padded_src_ids = pad_sequence(src_ids_batch, batch_first=True, padding_value=src_pad_idx)
    padded_tgt_input_ids = pad_sequence(tgt_input_ids_batch, batch_first=True, padding_value=tgt_pad_idx)
    padded_tgt_output_ids = pad_sequence(tgt_output_ids_batch, batch_first=True, padding_value=tgt_pad_idx)

    return {
        'src_ids': padded_src_ids,
        'tgt_input_ids': padded_tgt_input_ids,
        'tgt_output_ids': padded_tgt_output_ids
    }

batch_size = 2 # Using a small batch size for synthetic data

# Create a DataLoader instance
train_dataloader = DataLoader(
    translation_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)