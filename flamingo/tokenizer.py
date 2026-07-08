class DummyTokenizer:
    def __init__(self):
        self.vocab = {'[PAD]': 0, '[IMG]': 1, 'hello': 2, 'world': 3, 'this': 4, 'is': 5, 'a': 6, 'test': 7, 'image': 8, 'caption': 9, 'another': 10, 'example': 11, 'multimodal': 12, 'data': 13, 'the':14, 'third':15, 'with':16, '[UNK]': 17}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.img_token_id = self.vocab['[IMG]']
        self.pad_token_id = self.vocab['[PAD]']
        self.unk_token_id = self.vocab['[UNK]']
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        tokens = [self.vocab.get(word.lower(), self.unk_token_id) for word in text.split()]
        return tokens

    def pad(self, token_ids, max_len):
        if len(token_ids) > max_len:
            return token_ids[:max_len]
        return token_ids + [self.pad_token_id] * (max_len - len(token_ids))

    def get_img_token_id(self):
        return self.img_token_id


