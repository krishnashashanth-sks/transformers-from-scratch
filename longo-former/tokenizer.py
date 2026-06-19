import torch
import collections

class SimpleWordTokenizer:
  def __init__(self,):
    self.vocab={}
    self.id_to_token={}
    self.token_to_id={}
    self.special_tokens={
        '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[UNK]': 3
    }
    self._init_special_tokens()
  def _init_special_tokens(self):
    for token,id_val in self.special_tokens.items():
      self.token_to_id[token]=id_val
      self.id_to_token[id_val]=token
  def build_vocabulary(self,corpus):
    vocab_idx=len(self.special_tokens)
    word_counts=collections.defaultdict(int)
    for text in corpus:
      for word in text.lower().split():
        word_counts[word]+=1
    for word in sorted(word_counts.keys()):
      if word not in self.token_to_id:
        self.token_to_id[word]=vocab_idx
        self.id_to_token[vocab_idx]=word
        vocab_idx+=1
    self.vocab=self.token_to_id
    print(f"Vocabulary built with {len(self.vocab)} tokens.")
    print(f"Special tokens: {self.special_tokens}")
  def tokenize(self,text):
    return text.lower().split()
  def encode(self, text, max_length, global_attention_indices=None):
    tokens = self.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # 1. Convert to IDs
    input_ids = [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]

    # 2. Handle Truncation and Padding (Critical Fix)
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        # Everything in the truncated list is a real token
        attention_mask = [1] * max_length
    else:
        actual_len = len(input_ids)
        padding_length = max_length - actual_len
        # 1 for real tokens, 0 for [PAD]
        attention_mask = [1] * actual_len + [0] * padding_length
        input_ids = input_ids + [self.token_to_id['[PAD]']] * padding_length

    # 3. Handle Token Types and Global Attention
    token_type_ids = [0] * max_length
    global_attention_mask = torch.zeros(max_length, dtype=torch.long)

    if global_attention_indices is not None:
        for idx in global_attention_indices:
            if idx < max_length:
                global_attention_mask[idx] = 1

    # Standard CLS global attention
    if max_length > 0 and input_ids[0] == self.token_to_id['[CLS]']:
        global_attention_mask[0] = 1

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'global_attention_mask': global_attention_mask
    }