import torch

class SimulatedTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # Simple mapping for demonstration
        self.id_to_token = {i: f"[TOK{i}]" for i in range(vocab_size)}
        self.token_to_id = {f"[TOK{i}]": i for i in range(vocab_size)}

    def encode(self, text):
        # For simplicity, just return random IDs for example
        return [torch.randint(0, self.vocab_size, (1,)).item() for _ in range(len(text.split()))]

    def decode(self, token_ids):
        return " ".join([self.id_to_token.get(i, "[UNK]") for i in token_ids])
