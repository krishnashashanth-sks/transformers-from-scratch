import torch
from torch.utils.data import Dataset
from utils import prepare_llama_causal_lm_input

class LlamaCausalLMDataset(Dataset):
    def __init__(self, sample_texts, max_seq_len, pad_id):
        self.sample_texts = sample_texts
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.sample_texts)

    def __getitem__(self, index):
        text = self.sample_texts[index]

        input_ids, labels, attention_mask = prepare_llama_causal_lm_input(
            text, self.max_seq_len, self.pad_id
        )

        # Convert to torch.LongTensor
        input_ids_tensor = torch.LongTensor(input_ids)
        labels_tensor = torch.LongTensor(labels)
        # For Llama, the attention_mask from the dataset will be a padding mask.
        # The causal mask is generated internally by the model.
        attention_mask_tensor = torch.LongTensor(attention_mask)

        return {
            'input_ids': input_ids_tensor,
            'labels': labels_tensor,
            'attention_mask': attention_mask_tensor
        }