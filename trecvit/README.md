# TRecViT (Temporal Recurrent Vision Transformer)

TRecViT is a lightweight, educational implementation of a video-level Transformer architecture that combines:
- a 3D patch embedding for extracting spatial patches per frame,
- a small recurrent backbone (GRU) for temporal modeling,
- a Transformer encoder for global attention over the spatio-temporal token sequence,
- a classification head for video-level predictions.

This module is part of the "transformers-from-scratch" project and is intended for experimentation and learning rather than production use.

## Repository layout (trecvit/)
- model.py — High-level TRecViT module that composes the patch embedding, recurrent module, transformer encoder, and output head.
- layers.py — Building blocks:
  - VideoPatchEmbedding: Conv3d-based patch extraction + spatial + temporal positional embeddings + CLS token.
  - RecurrentModule: GRU temporal encoder (batch_first=True).
  - TransformerEncoderBlock / TransformerEncoder: standard pre-norm transformer blocks using nn.MultiheadAttention.
- dataset.py — Example dataset pipeline:
  - Downloads / expects a local mp4, extracts frames, builds a VideoDataset that yields clips of fixed sequence_length (default 16).
  - transforms_pipeline for resizing, ToTensor and normalization.
  - Produces a PyTorch DataLoader with samples shaped for the model: (batch_size, C, T, H, W).
- train.py — train_model(...) function: runs epochs, computes loss/accuracy, updates optimizer and scheduler, returns training history.
- inference.py — predict_class_label(...) helper that runs evaluation and returns predicted label + confidence.
- main.py — example script assembling hyperparameters, model, loss, optimizer, scheduler, calling train_model, and running a single inference demo.

## Quick concepts / expected tensor shapes
- Video tensor format used across the code: (batch_size, channels, num_frames, height, width) — i.e., (B, C, T, H, W).
- VideoPatchEmbedding:
  - Uses Conv3d with kernel_size=(1, patch_h, patch_w) and stride=(1, patch_h, patch_w) so patches are extracted per frame (temporal dimension preserved).
  - Output tokens (after spatial+temporal embeddings) are flattened to shape (B, seq_len, embed_dim) where seq_len = 1 (CLS) + T * num_spatial_patches.
- RecurrentModule uses a GRU and returns (B, seq_len, hidden_dim) — in this repo hidden_dim == embed_dim to keep dimensions consistent for the Transformer.
- TransformerEncoder expects (B, seq_len, embed_dim) as input (layers in layers.py handle the attention permutation internally).

## Minimal requirements
- Python 3.8+
- PyTorch
- torchvision
- einops
- opencv-python (cv2) — for frame extraction scripts in dataset.py
- Pillow
- tqdm

Install with:
```bash
pip install torch torchvision einops opencv-python pillow tqdm
```

## Example: training (high-level)
The included example flow (main.py) demonstrates:
1. Create model:
   - image_size=(128,128), patch_size=(16,16), embed_dim=768, num_frames=16, num_transformer_layers=6, num_heads=12, num_classes=10
2. Define loss & optimizer:
   - CrossEntropyLoss(), AdamW(lr=1e-4)
3. Device:
   - model.to(device) where device = "cuda" if available
4. Train:
   - history = train_model(model, dataloader, loss_fn, optimizer, scheduler, device, num_epochs)

Notes:
- main.py expects a variable `dataloader` exposed by dataset.py and `num_epochs` to be set; adapt these to your experiment.
- The example hyperparameters are typical transformer defaults but can be lowered for faster experiments (e.g., smaller embed_dim, fewer layers).

Example run (after preparing dataset and dataloader):
```bash
python trecvit/main.py
```
(You may need to adjust main.py to parse CLI args for num_epochs, data paths, etc.)

## Dataset / data preparation
- dataset.py demonstrates:
  - Downloading a sample MP4 (placeholder), extracting frames to `UCF101_frames/<video_name>/frame_XXXXX.jpg`.
  - VideoDataset class that creates clips of fixed `sequence_length` (default 16) with half-overlap.
  - transforms_pipeline: Resize((128,128)), ToTensor(), Normalize(...) — results stacked to (C, T, H, W).
- For real experiments, prepare a dataset with labelled subdirectories or annotation files and adapt VideoDataset to read labels per-video.

Important: dataset.py is a demo script and will attempt to write / create directories in the current working directory. Verify paths before running.

## Training details
- train.py provides a straightforward epoch loop:
  - For each batch: forward -> loss -> backward -> optimizer.step()
  - Tracks epoch loss, accuracy, and scheduler-adjusted LR.
  - Uses tqdm.notebook.tqdm for progress bars (works in Jupyter/Colab; change if running in plain terminal).

## Inference
- inference.py: predict_class_label(model, video_tensor, idx_to_class, device)
  - Accepts a single video tensor (C, T, H, W) or batched (B, C, T, H, W)
  - Returns (predicted_label_str, confidence_float)

## Suggested improvements and TODOs
- Add CLI arg parsing to main.py (argparse or Hydra) to set dataset paths, hyperparams, checkpoints, and num_epochs.
- Implement checkpoint save/load (torch.save / torch.load) during training for resume and best-model saving.
- Replace placeholder dataset download with proper dataset handling (UCF-101, Kinetics, or custom dataset) with labels.
- Add unit tests for shapes and small forward/backward passes.
- Add documentation for expected memory usage and tips for training on limited GPUs (reduce batch size, embed_dim, layers).

## Example usage snippets

- Instantiate and run a single forward pass:
```python
from trecvit.model import TRecViT
import torch

model = TRecViT(
    image_size=(128,128), patch_size=(16,16), in_channels=3,
    embed_dim=128, num_frames=16, num_gru_layers=1,
    num_attention_heads=4, mlp_dim=512, num_transformer_layers=2,
    num_classes=10, dropout_rate=0.1
)
dummy_input = torch.randn(2, 3, 16, 128, 128)  # (B, C, T, H, W)
logits = model(dummy_input)  # (2, num_classes)
```

- Run training (conceptual):
```python
from trecvit.train import train_model
history = train_model(model, dataloader, torch.nn.CrossEntropyLoss(), optimizer, scheduler, device, num_epochs=5)
```

## License & Attribution
- See the root repository for license information. This module is for learning/educational purposes.
