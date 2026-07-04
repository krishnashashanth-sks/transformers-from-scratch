corpus = "the quick brown fox jumps over the lazy dog dog dog quick quick brown fox brown"

bpe_tokenizer = BasicBPETokenizer()
bpe_tokenizer.train(corpus, num_merges=10)

print("\nLearned Merges:")
for pair, new_token in bpe_tokenizer.merges.items():
    print(f"  {pair} -> {new_token}")

print("\nFinal Vocabulary (Token to ID mapping):")
for token, idx in sorted(bpe_tokenizer.token_to_id.items(), key=lambda item: item[1]):
    print(f"  {token}: {idx}")
    
text_to_encode = "the quick brown fox"
encoded_ids = bpe_tokenizer.encode(text_to_encode)
print(f"\nOriginal text: '{text_to_encode}'")
print(f"Encoded IDs: {encoded_ids}")

decoded_text = bpe_tokenizer.decode(encoded_ids)
print(f"Decoded text: '{decoded_text}'")

text_with_new_word = "a quick brown cat"
encoded_ids_new = bpe_tokenizer.encode(text_with_new_word)
print(f"\nOriginal text (new word): '{text_with_new_word}'")
print(f"Encoded IDs (new word): {encoded_ids_new}")
decoded_text_new = bpe_tokenizer.decode(encoded_ids_new)
print(f"Decoded text (new word): '{decoded_text_new}'")
