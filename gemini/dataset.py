from torch.utils.data import Dataset
import torch

# Dummy Dataset and DataLoader
class DummyMultimodalDataset(Dataset):
    def __init__(self, num_samples, text_vocab_size, text_max_seq_len,
                 image_size, audio_n_mels, audio_len, video_frames,
                 num_classes, embed_dim):
        self.num_samples = num_samples
        self.text_vocab_size = text_vocab_size
        self.text_max_seq_len = text_max_seq_len
        self.image_size = image_size
        self.audio_n_mels = audio_n_mels
        self.audio_len = audio_len
        self.video_frames = video_frames
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy inputs
        text_tokens = torch.randint(0, self.text_vocab_size, (self.text_max_seq_len,))
        images = torch.randn(3, self.image_size, self.image_size)
        audio_spectrograms = torch.randn(self.audio_n_mels, self.audio_len)
        videos = torch.randn(self.video_frames, 3, self.image_size, self.image_size)

        # Dummy targets for various tasks
        text_targets = torch.randint(0, self.text_vocab_size, (self.text_max_seq_len,))
        image_targets = torch.randint(0, self.num_classes, (1,)).squeeze(0)
        audio_targets = torch.randint(0, self.num_classes, (1,)).squeeze(0)
        video_targets = torch.randint(0, self.num_classes, (1,)).squeeze(0)

        # Dummy alignment embeddings (e.g., for contrastive loss)
        multimodal_alignment_embeddings_a = torch.randn(self.embed_dim)
        multimodal_alignment_embeddings_b = torch.randn(self.embed_dim)

        return {
            'text_tokens': text_tokens,
            'images': images,
            'audio_spectrograms': audio_spectrograms,
            'videos': videos,
            'text_targets': text_targets,
            'image_targets': image_targets,
            'audio_targets': audio_targets,
            'video_targets': video_targets,
            'multimodal_alignment_embeddings_a': multimodal_alignment_embeddings_a,
            'multimodal_alignment_embeddings_b': multimodal_alignment_embeddings_b,
        }


