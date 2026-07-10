import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange # A library for more readable tensor manipulations

# Helper function for a simple FeedForward network (FFN)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Multi-Head Self-Attention (MHSA)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#### Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

#### Mixture-of-Experts (MoE) Layer
class MoE(nn.Module):
    def __init__(self, dim, num_experts, hidden_dim, top_k=2, dropout=0.):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network (router)
        self.gate = nn.Linear(dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

        # Expert networks (FeedForward modules)
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, dropout) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Get raw gate scores
        gate_scores = self.gate(x_flat)

        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = self.softmax(top_k_scores) # Apply softmax to scores

        # Initialize output and expert outputs
        output = torch.zeros_like(x_flat)
        expert_outputs = torch.zeros(batch_size * seq_len, self.num_experts, dim, device=x.device)

        # Route input to experts
        for i, expert in enumerate(self.experts):
            expert_outputs[:, i, :] = expert(x_flat)

        # Combine outputs from top-k experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dim)
            output += torch.gather(expert_outputs, 1, expert_idx).squeeze(1) * top_k_scores[:, i].unsqueeze(-1)

        # Reshape back to original sequence dimensions
        return output.view(batch_size, seq_len, dim)

#### Multimodal Input Embedding Layers
class TextEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim) # Learned positional embeddings
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_embeddings(tokens) + self.position_embeddings(positions)
        return self.dropout(x)

class ImageEmbeddingLayer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) # +1 for CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # Rearrange image into patches
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = self.patch_embedding(x)

        # Add CLS token and positional embeddings
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embeddings
        return self.dropout(x)

class AudioEmbeddingLayer(nn.Module):
    def __init__(self, audio_len, sample_rate, n_mels, patch_size, embed_dim, dropout=0.):
        super().__init__()
        # Simplified: assume audio is pre-processed into mel spectrograms
        # audio_len: number of time steps in spectrogram
        # n_mels: number of mel frequency bins

        assert n_mels % patch_size == 0, 'Mel frequency bins (height) must be divisible by patch size'
        assert audio_len % patch_size == 0, 'Audio length (width) must be divisible by patch size'
        num_patches = (n_mels // patch_size) * (audio_len // patch_size) # 2D patch on spectrogram
        patch_dim = patch_size * patch_size # Treating spectrogram as an image essentially, with 1 implicit channel

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_spectrogram):
        # Assuming audio_spectrogram is (batch, n_mels, audio_len)
        # Rearrange spectrogram into 2D patches (height=n_mels, width=audio_len)
        x = rearrange(audio_spectrogram, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(x)
        x += self.position_embeddings
        return self.dropout(x)

class VideoEmbeddingLayer(nn.Module):
    def __init__(self, video_frames, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()
        # Video is treated as a sequence of images + temporal dimension
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_spatial_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2 # Each spatial patch dim

        self.patch_size = patch_size
        self.video_frames = video_frames
        self.image_embedder = nn.Linear(patch_dim, embed_dim) # Embed each spatial patch
        # Spatio-temporal positional embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, video_frames * num_spatial_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, video):
        # video: (batch, frames, channels, height, width)
        batch, frames, c, h, w = video.shape

        # Flatten frames and patch them
        x = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = self.image_embedder(x)

        x += self.position_embeddings # Add spatio-temporal positional embeddings
        return self.dropout(x)

#### Output Heads
class TextGenerationHead(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, transformer_output_text_tokens): # Assuming text tokens are at a known slice of the output
        return self.proj(transformer_output_text_tokens)

class MultimodalClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.proj = nn.Linear(embed_dim, num_classes)

    def forward(self, transformer_output_cls_token): # Assuming a CLS token or pooled representation
        return self.proj(transformer_output_cls_token)

# Example: Tool Use/Function Calling Head (conceptual, would output structured data)
class ToolUseHead(nn.Module):
    def __init__(self, embed_dim, num_tools, max_arg_tokens):
        super().__init__()
        self.tool_classifier = nn.Linear(embed_dim, num_tools) # Predict which tool to use
        self.arg_generator = nn.Linear(embed_dim, max_arg_tokens * 100) # Simplified: predicts flat arg vector
        # In a real scenario, arg_generator would be a seq-to-seq decoder generating JSON/code

    def forward(self, transformer_output_cls_token):
        tool_logits = self.tool_classifier(transformer_output_cls_token)
        args_representation = self.arg_generator(transformer_output_cls_token)
        return tool_logits, args_representation
