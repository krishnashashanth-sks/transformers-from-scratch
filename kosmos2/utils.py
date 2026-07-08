# A dummy text tokenizer for the example
def dummy_text_tokenizer(text: str) -> list[int]:
    token_map = {'a':10, 'cat':11, 'sits':12, 'on':13, 'mat':14, 'the':15, '.':16, 'this':17, 'box':18}
    tokens = [token_map.get(word.lower(), 0) for word in text.split()] # 0 for unknown words
    return tokens

# Helper function for identity or gelu activation
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

