import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TextGenerationHead,MultimodalClassificationHead

# 1. & 2. Text Loss (for text generation/language modeling)
class TextLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        # Reshape for CrossEntropyLoss: (N, C, ...) where C is vocab_size
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        logits = logits.permute(0, 2, 1) # (batch_size, vocab_size, seq_len)
        return self.loss_fn(logits, targets)

# 3. Image, Audio, Video Loss (conceptual, for classification or masked reconstruction)
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

class ImageLoss(ClassificationLoss):
    pass

class AudioLoss(ClassificationLoss):
    pass

class VideoLoss(ClassificationLoss):
    pass

# 4. Cross-Modal Alignment Loss (conceptual InfoNCE-like loss)
class CrossModalAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_a, embeddings_b):
        # embeddings_a and embeddings_b are feature vectors for aligned pairs
        # For example, embeddings_a could be from image, embeddings_b from text description of the image.
        # We assume they are (batch_size, embed_dim)

        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Calculate similarity scores
        # Similarities from A to B
        logits_ab = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature
        # Similarities from B to A
        logits_ba = torch.matmul(embeddings_b, embeddings_a.T) / self.temperature

        # Create labels for self-supervised contrastive learning (diagonal is positive)
        labels = torch.arange(logits_ab.shape[0], device=logits_ab.device)

        loss_a = F.cross_entropy(logits_ab, labels)
        loss_b = F.cross_entropy(logits_ba, labels)

        return (loss_a + loss_b) / 2

class CombinedLoss(nn.Module):
    def __init__(self,
                 text_loss_weight=1.0,
                 image_loss_weight=1.0,
                 audio_loss_weight=1.0,
                 video_loss_weight=1.0,
                 alignment_loss_weight=1.0,
                 text_vocab_size=None, # Required for text_gen_head and text_loss
                 num_classes=None, # Required for classification heads
                 embed_dim=None
                ):
        super().__init__()
        # Initialize individual loss functions
        self.text_loss_fn = TextLoss()
        self.image_loss_fn = ImageLoss()
        self.audio_loss_fn = AudioLoss()
        self.video_loss_fn = VideoLoss()
        self.alignment_loss_fn = CrossModalAlignmentLoss()

        # Store weights
        self.weights = {
            'text': text_loss_weight,
            'image': image_loss_weight,
            'audio': audio_loss_weight,
            'video': video_loss_weight,
            'alignment': alignment_loss_weight,
        }

        # Instantiate output heads for conceptual usage within CombinedLoss
        # In a real model, these would be part of the model itself.
        # This is for demonstrating how losses are calculated with outputs.
        if text_vocab_size is not None and embed_dim is not None:
            self.text_gen_head = TextGenerationHead(embed_dim, text_vocab_size)
        else:
            self.text_gen_head = None

        if num_classes is not None and embed_dim is not None:
            self.multimodal_clf_head = MultimodalClassificationHead(embed_dim, num_classes)
        else:
            self.multimodal_clf_head = None

    def forward(self,
                model_output,
                text_targets=None,
                image_targets=None,
                audio_targets=None,
                video_targets=None,
                text_max_seq_len=None, # To slice text_output from model_output
                multimodal_alignment_embeddings_a=None,
                multimodal_alignment_embeddings_b=None
               ):

        total_loss = torch.tensor(0.0, device=model_output.device)
        losses = {}

        # --- Text Loss ---
        if self.text_gen_head and text_targets is not None and text_max_seq_len is not None:
            # Assume text tokens are the first part of the sequence output
            text_output_for_gen = model_output[:, :text_max_seq_len, :]
            text_logits = self.text_gen_head(text_output_for_gen)
            text_loss = self.text_loss_fn(text_logits, text_targets)
            total_loss += self.weights['text'] * text_loss
            losses['text'] = text_loss.item()

        # --- Image, Audio, Video Classification Losses ---
        # For classification, usually a CLS token or pooled representation is used
        # We'll assume the first token of the model_output is the CLS token for classification
        cls_output_for_clf = model_output[:, 0, :]

        if self.multimodal_clf_head and image_targets is not None:
            image_logits = self.multimodal_clf_head(cls_output_for_clf) # Simplified: all use same head
            image_loss = self.image_loss_fn(image_logits, image_targets)
            total_loss += self.weights['image'] * image_loss
            losses['image'] = image_loss.item()

        if self.multimodal_clf_head and audio_targets is not None:
            audio_logits = self.multimodal_clf_head(cls_output_for_clf) # Simplified: all use same head
            audio_loss = self.audio_loss_fn(audio_logits, audio_targets)
            total_loss += self.weights['audio'] * audio_loss
            losses['audio'] = audio_loss.item()

        if self.multimodal_clf_head and video_targets is not None:
            video_logits = self.multimodal_clf_head(cls_output_for_clf) # Simplified: all use same head
            video_loss = self.video_loss_fn(video_logits, video_targets)
            total_loss += self.weights['video'] * video_loss
            losses['video'] = video_loss.item()

        # --- Cross-Modal Alignment Loss ---
        if multimodal_alignment_embeddings_a is not None and multimodal_alignment_embeddings_b is not None:
            alignment_loss = self.alignment_loss_fn(multimodal_alignment_embeddings_a, multimodal_alignment_embeddings_b)
            total_loss += self.weights['alignment'] * alignment_loss
            losses['alignment'] = alignment_loss.item()

        return total_loss, losses
