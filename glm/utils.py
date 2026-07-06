import random
import torch

def generate_synthetic_text(num_samples, min_words, max_words, vocab):
    data = []
    for _ in range(num_samples):
        num_words = random.randint(min_words, max_words)
        sentence = ' '.join(random.choice(vocab) for _ in range(num_words))
        data.append(sentence)
    return data


def preprocess_text(text, tokenizer, max_len):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    # Truncate or pad
    if len(ids) > max_len:
        ids = ids[:max_len]
    elif len(ids) < max_len:
        padding_length = max_len - len(ids)
        ids = ids + [tokenizer.pad_id] * padding_length

    # Create attention mask
    # 1 for real tokens, 0 for padding tokens
    attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    # Ensure attention mask is also truncated if ids were truncated
    attention_mask = attention_mask[:max_len]

    return torch.tensor(ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

# Test the preprocessing function
