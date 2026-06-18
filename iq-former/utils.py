import torch

def tokenize_sentence(sentence):
    """Simple tokenizer: splits a sentence into a list of words."""
    return [word.lower() for word in sentence.split()] # Convert to lowercase for consistency

def tokens_to_ids(tokens, vocab):
    """Converts a list of tokens to a numerical sequence using the vocabulary."""
    return [vocab.word2idx.get(token, vocab.word2idx['<unk>']) for token in tokens]

def ids_to_tokens(ids, vocab):
    """Converts a numerical sequence to a list of tokens using the vocabulary."""
    return [vocab.idx2word.get(idx, '<unk>') for idx in ids]

def make_src_mask(src, pad_idx):
    """Create a mask for source padding."""
    # (batch_size, 1, 1, seq_len)
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_idx):
    """Create a mask for target padding and subsequent words."""
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
    tgt_seq_len = tgt.size(-1)

    # Creates a square matrix where the upper triangle (future words) is True
    # and the lower triangle (past and current words) is False.
    # unsqueeze(0) for batch dimension, unsqueeze(1) for head dimension
    tgt_nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len, device=tgt.device), diagonal=1)).bool()

    # Combine the padding mask and the subsequent word mask
    # Both masks are 1s where attention is allowed, 0s where it's not.
    # So we want to keep attention only where both are 1.
    return tgt_pad_mask & tgt_nopeak_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, src_pad_idx, tgt_pad_idx, device):
    """Greedily decodes a sequence token by token."""
    # Ensure source input is on the correct device
    src = src.to(device)
    src_mask = src_mask.to(device)

    # Encode the source sequence
    memory = model.encode(src, src_mask)

    # Initialize the target sequence with the start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)

    for i in range(max_len - 1):
        # Create target mask for the current generated sequence
        # tgt_mask needs to be (1, 1, ys.size(1), ys.size(1)) for a single example
        # The make_tgt_mask function expects a batch_size dimension at the start
        tgt_mask = make_tgt_mask(ys, tgt_pad_idx)

        # Decode one step further
        out = model.decode(memory, src_mask, ys, tgt_mask)

        # Pass the output for the last token through the generator to get log probabilities
        prob = model.generator(out[:, -1])

        # Predict the next token (greedy approach: take the argmax)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Concatenate the predicted token to the current target sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).to(device)], dim=1)

        # If the predicted token is the end symbol, break the loop
        if next_word == end_symbol:
            break
    return ys