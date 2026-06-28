import random

def mask_tokens(token_ids, vocab, mask_token_id, cls_token_id, sep_token_id, pad_token_id, mask_prob=0.15):
    # Create a copy of token_ids for modification
    masked_token_ids = list(token_ids)

    # Initialize labels with PAD_ID, indicating non-masked tokens
    # Only actual masked positions will have their original token ID
    labels = [pad_token_id] * len(token_ids)

    # Get a list of possible tokens for random replacement (excluding special tokens if desired,
    # but for simplicity, we can sample from all valid vocab IDs and let the model learn)
    # For more robust implementation, one would filter out special tokens from vocab.values()
    # Let's consider a simple sampling approach for now.
    # For this basic implementation, we'll sample from all non-special vocab IDs for simplicity.
    # However, a proper BERT implementation samples from the entire vocabulary.

    # Prepare a list of valid token IDs for random replacement (excluding special tokens)
    # The actual BERT implementation samples from the full vocabulary, but if we want to avoid
    # special tokens appearing as random replacements, we can filter them.
    # For this exercise, let's keep it simple and assume sampling from the full vocab is fine.
    # vocab_size is already defined from previous steps
    all_vocab_ids = list(range(len(vocab)))

    for i, token_id in enumerate(token_ids):
        # Do not mask special tokens like [CLS] or [SEP]
        if token_id == cls_token_id or token_id == sep_token_id:
            continue

        # Randomly decide if this token should be masked
        if random.random() < mask_prob:
            # This token will be masked, so store its original ID as a label
            labels[i] = token_id

            # Apply masking strategy (80% [MASK], 10% random, 10% original)
            rand_num = random.random()
            if rand_num < 0.8:  # 80% of the time: replace with [MASK] token
                masked_token_ids[i] = mask_token_id
            elif rand_num < 0.9: # 10% of the time: replace with a random token
                # Ensure the random token is not a special token itself if desired,
                # but for this basic example, we'll just pick a random vocab ID.
                masked_token_ids[i] = random.choice(all_vocab_ids)
            # else (10% of the time): keep the original token (masked_token_ids[i] remains token_id)

    return masked_token_ids, labels

def tokenize(text,vocab):
    # Convert to lowercase and split into words
    # Split by spaces but also handle special tokens separately
    words = []
    temp_words = text.lower().replace('.', '').split()
    for word in temp_words:
        if word == '[mask]': # Recognize the special token string
            words.append('[MASK]')
        else:
            words.append(word)

    # Convert words to token IDs, handle unknown words
    token_ids = []
    for word in words:
        token_ids.append(vocab.get(word, vocab['[UNK]']))
    return token_ids


def create_nsp_pairs(sample_texts, num_pairs):
    nsp_data = []
    num_sentences = len(sample_texts)

    if num_sentences < 2: # Need at least two sentences to form any pair
        print("Not enough sentences to create NSP pairs.")
        return nsp_data

    # Pre-tokenize all sentences to avoid re-tokenizing in the loop
    tokenized_sentences = [tokenize(text) for text in sample_texts]

    for _ in range(num_pairs):
        # Randomly decide whether to create a consecutive pair (is_next=True) or a random pair (is_next=False)
        if random.random() < 0.5:  # Approximately 50% for is_next=True
            # is_next=True pair
            # Ensure there's a next sentence available
            if num_sentences < 2: # Edge case if list somehow becomes too small
                continue
            idx_a = random.randrange(num_sentences - 1) # Ensure idx_a + 1 is valid
            sentence_a = sample_texts[idx_a]
            sentence_b = sample_texts[idx_a + 1]
            is_next_label = 1
        else:
            # is_next=False pair
            idx_a = random.randrange(num_sentences)
            sentence_a = sample_texts[idx_a]

            # Find a sentence_b that is not consecutive and not the same as sentence_a
            possible_idx_b = list(range(num_sentences))
            # Remove idx_a and idx_a + 1 (if valid) from possibilities for idx_b
            if idx_a in possible_idx_b:
                possible_idx_b.remove(idx_a)
            if idx_a + 1 < num_sentences and (idx_a + 1) in possible_idx_b:
                possible_idx_b.remove(idx_a + 1)

            if not possible_idx_b: # If no valid non-consecutive sentence can be found (e.g., only 2 sentences total)
                # Fallback: choose a random sentence B, might be consecutive or same if no other choice
                idx_b = random.randrange(num_sentences)
            else:
                idx_b = random.choice(possible_idx_b)

            sentence_b = sample_texts[idx_b]
            is_next_label = 0

        nsp_data.append((sentence_a, sentence_b, is_next_label))

    return nsp_data

max_seq_len = 50 # Define a constant for maximum sequence length

def prepare_bert_input(token_ids_a, token_ids_b, max_seq_len, cls_id, sep_id, pad_id):
    # 3. Construct the input sequence
    input_ids = [cls_id] + token_ids_a + [sep_id]

    # 4. Create segment IDs for sentence A
    segment_ids = [0] * (len(token_ids_a) + 2) # +2 for [CLS] and [SEP]

    if token_ids_b is not None:
        input_ids += token_ids_b + [sep_id]
        segment_ids += [1] * (len(token_ids_b) + 1) # +1 for [SEP]

    # 5. Handle truncation and padding
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        segment_ids = segment_ids[:max_seq_len]
    else:
        padding_len = max_seq_len - len(input_ids)
        input_ids += [pad_id] * padding_len
        segment_ids += [0] * padding_len # Padding tokens usually get segment ID 0 or are ignored.

    # 6. Generate attention mask
    attention_mask = [1] * (max_seq_len - padding_len) + [0] * padding_len

    return input_ids, segment_ids, attention_mask