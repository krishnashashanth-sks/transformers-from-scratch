import torch

# Helper function to create a dummy tokenizer (for demonstration purposes)
class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, return_tensors='pt'):
        # Simulate encoding a prompt
        # Ensure the token_ids are within the vocab_size range [0, vocab_size-1]
        tokens = torch.randint(2, self.vocab_size, (1, len(text.split())), dtype=torch.long) # Random tokens, avoiding pad/eos
        if return_tensors == 'pt':
            return tokens
        return tokens.tolist()[0]

    def decode(self, token_ids, skip_special_tokens=True):
        # Simulate decoding tokens back to text
        # Ensure that only valid tokens are decoded, ignoring pad/eos
        return " ".join([str(t) for t in token_ids if t not in [self.pad_token_id, self.eos_token_id]])

