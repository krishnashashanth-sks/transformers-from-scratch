class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [hash(word) % self.vocab_size for word in text.split()]

    def decode(self, token_ids):
        # Simple string representation of token IDs
        return " ".join(map(str, token_ids))