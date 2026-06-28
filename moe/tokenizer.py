import torch
import collections

# --- Dummy Tokenizer for Demonstration ---
class DummyTokenizer:
    def __init__(self, vocab_size, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):
        self.vocab_size = vocab_size
        self.word_to_id = {
            pad_token: 0,
            unk_token: 1,
            eos_token: 2,
        }
        self.id_to_word = {0: pad_token, 1: unk_token, 2: eos_token}
        current_id = 3
        # Add some dummy words
        for i in range(vocab_size - 3):
            word = f'word_{i}'
            self.word_to_id[word] = current_id
            self.id_to_word[current_id] = word
            current_id += 1

        self.pad_token_id = self.word_to_id[pad_token]
        self.unk_token_id = self.word_to_id[unk_token]
        self.eos_token_id = self.word_to_id[eos_token]

    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(word, self.unk_token_id) for word in words]

    def decode(self, token_ids):
        return ' '.join([self.id_to_word.get(idx, '<unk>') for idx in token_ids if idx not in [self.pad_token_id]])
