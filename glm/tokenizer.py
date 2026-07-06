class BasicTokenizer:
    def __init__(self, vocab):
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        # Add special tokens to vocabulary if not already present
        if self.unk_token not in self.word_to_id:
            self.word_to_id[self.unk_token] = len(self.word_to_id)
            self.id_to_word[len(self.id_to_word)] = self.unk_token
        if self.pad_token not in self.word_to_id:
            self.word_to_id[self.pad_token] = len(self.word_to_id)
            self.id_to_word[len(self.id_to_word)] = self.pad_token

        self.unk_id = self.word_to_id[self.unk_token]
        self.pad_id = self.word_to_id[self.pad_token]
        self.vocab_size = len(self.word_to_id)

    def tokenize(self, text):
        return text.split(' ')

    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_id.get(token, self.unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_word.get(id, self.unk_token) for id in ids]