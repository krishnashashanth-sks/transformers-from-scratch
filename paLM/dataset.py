from torch.utils.data import Dataset
import torch

# --- Custom Dataset Definition ---
class ConceptualMultimodalDataset(Dataset):
    def __init__(self, text_samples, image_feature_dim, num_image_tokens, max_text_seq_len, output_dim, tokenizer_func,tokenizer):
        self.text_samples = text_samples
        self.image_feature_dim = image_feature_dim
        self.num_image_tokens = num_image_tokens
        self.max_text_seq_len = max_text_seq_len
        self.output_dim = output_dim
        self.tokenizer_func = tokenizer_func

        # Tokenize all text samples once
        self.tokenized_texts = self.tokenizer_func(text_samples,tokenizer, max_length=self.max_text_seq_len)

        # Generate dummy image features and target labels for all samples
        # In a real scenario, these would be loaded from actual data.
        self.dummy_image_features = torch.rand(len(text_samples), self.num_image_tokens, self.image_feature_dim)
        
        combined_seq_len = self.tokenized_texts.size(1) + self.num_image_tokens
        self.dummy_target_labels = torch.randint(0, self.output_dim, (len(text_samples), combined_seq_len))

    def __len__(self):
        return len(self.text_samples)

    def __getitem__(self, idx):
        text_tokens = self.tokenized_texts[idx]
        image_features = self.dummy_image_features[idx]
        target_labels = self.dummy_target_labels[idx]
        return text_tokens, image_features, target_labels

