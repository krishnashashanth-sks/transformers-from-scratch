from collections import defaultdict, Counter

class BasicBPETokenizer:
    def __init__(self, vocab_size=None, special_tokens=None):
        self.vocab_size = vocab_size
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.initial_vocab = set()
        self.special_tokens = special_tokens if special_tokens is not None else []

    def _get_vocab(self, text):
        words = text.split(' ')
        vocab = Counter()
        for word in words:
            for char in word:
                vocab[char] += 1
            if word:
                vocab[' '] += 1
        return vocab

    def _count_pairs(self, tokens):
        pairs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        result = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def train(self, text_corpus, num_merges=100):
        words_split_by_char = []
        for word in text_corpus.split():
            words_split_by_char.append(list(word) + ['</w>'])

        initial_tokens = sorted(list(set(char for word_chars in words_split_by_char for char in word_chars) | set(self.special_tokens)))
        self.token_to_id = {token: i for i, token in enumerate(initial_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(initial_tokens)}
        next_id = len(initial_tokens)

        current_tokens_per_word = {tuple(word_chars): count for word_chars, count in Counter(tuple(w) for w in words_split_by_char).items()}

        for i in range(num_merges):
            pairs = defaultdict(int)
            for word_tokens, count in current_tokens_per_word.items():
                word_pairs = self._count_pairs(word_tokens)
                for pair, pair_count in word_pairs.items():
                    pairs[pair] += pair_count * count

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)

            self.merges[best_pair] = new_token
            self.token_to_id[new_token] = next_id
            self.id_to_token[next_id] = new_token
            next_id += 1

            new_current_tokens_per_word = defaultdict(int)
            for word_tokens, count in current_tokens_per_word.items():
                merged_word_tokens = tuple(self._merge_pair(list(word_tokens), best_pair, new_token))
                new_current_tokens_per_word[merged_word_tokens] += count
            current_tokens_per_word = new_current_tokens_per_word

        print(f"Training complete. Vocabulary size: {len(self.token_to_id)}")

    def encode(self, text):
        tokens = []
        words = text.split()
        for word in words:
            word_tokens = list(word) + ['</w>']

            for pair, new_token in self.merges.items():
                word_tokens = self._merge_pair(word_tokens, pair, new_token)

            tokens.extend(word_tokens)

        encoded_ids = []
        for token in tokens:
            if token in self.token_to_id:
                encoded_ids.append(self.token_to_id[token])
            else:
                for char in token:
                    if char in self.token_to_id:
                        encoded_ids.append(self.token_to_id[char])
                    else:
                        pass
        return encoded_ids

    def decode(self, token_ids):
        decoded_tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                decoded_tokens.append(self.id_to_token[tid])
            else:
                pass

        text = ''.join(decoded_tokens).replace('</w>', ' ').strip()
        return text