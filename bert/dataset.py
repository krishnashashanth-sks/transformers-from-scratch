from utils import *
import torch
from torch.utils.data import Dataset, DataLoader

sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The sun rises in the east every morning.",
    "Water is essential for all living things.",
    "Artificial intelligence is changing the world.",
    "Machine learning algorithms can analyze vast amounts of data.",
    "Natural language processing helps computers understand human language.",
    "Deep learning models require significant computational resources.",
    "Python is a popular programming language for data science.",
    "TensorFlow and PyTorch are leading deep learning frameworks.",
    "The Earth orbits the sun in an elliptical path.",
    "Birds migrate south for the winter to find warmer climates.",
    "Books offer a window into different worlds and ideas.",
    "Coding helps develop problem-solving skills.",
    "Music can evoke a wide range of emotions and memories."
]

special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
vocab = {token: i for i, token in enumerate(special_tokens)}

mask_id = vocab['[MASK]']
cls_id = vocab['[CLS]']
sep_id = vocab['[SEP]']
pad_id = vocab['[PAD]']

# Extract all unique words from sample_texts and convert to lowercase
all_words = []
for text in sample_texts:
    all_words.extend(text.lower().replace('.', '').split())

unique_words = sorted(list(set(all_words)))

# Add unique words to the vocabulary, starting after special tokens
for word in unique_words:
    if word not in vocab:
        vocab[word] = len(vocab)

# Create an inverse mapping from ID to token
idx_to_token = {idx: token for token, idx in vocab.items()}

class BERTDataset(Dataset):
    def __init__(self, sample_texts, vocab, max_seq_len, num_nsp_pairs, cls_id, sep_id, mask_id, pad_id):
        self.sample_texts = sample_texts
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_id = mask_id
        self.pad_id = pad_id

        # Generate NSP examples
        self.nsp_examples = create_nsp_pairs(sample_texts, num_nsp_pairs)

    def __len__(self):
        return len(self.nsp_examples)

    def __getitem__(self, index):
        # 5a. Retrieve the sentence texts and NSP label
        sentence_a_text, sentence_b_text, nsp_label = self.nsp_examples[index]

        # 5b. Tokenize sentences
        token_ids_a = tokenize(sentence_a_text,vocab)
        token_ids_b = tokenize(sentence_b_text,vocab)

        # 5c. Prepare BERT input (combine, pad, create attention mask)
        full_sequence_input_ids, segment_ids, attention_mask = prepare_bert_input(
            token_ids_a, token_ids_b, self.max_seq_len, self.cls_id, self.sep_id, self.pad_id
        )

        # 5d. Apply MLM masking
        masked_input_ids, mlm_labels = mask_tokens(
            full_sequence_input_ids, self.vocab, self.mask_id, self.cls_id, self.sep_id, self.pad_id
        )

        # 5e. Convert to torch.LongTensor
        input_ids_tensor = torch.LongTensor(masked_input_ids)
        segment_ids_tensor = torch.LongTensor(segment_ids)
        attention_mask_tensor = torch.LongTensor(attention_mask)
        mlm_labels_tensor = torch.LongTensor(mlm_labels)
        nsp_labels_tensor = torch.LongTensor([nsp_label]) # NSP label is a single value

        # 5f. Return dictionary of tensors
        return {
            'input_ids': input_ids_tensor,
            'segment_ids': segment_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'mlm_labels': mlm_labels_tensor,
            'nsp_labels': nsp_labels_tensor
        }

num_nsp_pairs_for_dataset = 100 # More pairs for a larger dataset

bert_dataset = BERTDataset(
    sample_texts=sample_texts,
    vocab=vocab,
    max_seq_len=max_seq_len,
    num_nsp_pairs=num_nsp_pairs_for_dataset,
    cls_id=cls_id,
    sep_id=sep_id,
    mask_id=mask_id,
    pad_id=pad_id
)