# IQ-Former

A compact, from-scratch implementation of Transformer components and a small training pipeline using PyTorch. This module provides the core layers, model assembly, dataset utilities, training loop, and helper functions so you can study, extend, and run experiments without higher-level frameworks.

## Repository layout (iq-former/)
- `dataset.py` — Dataset and data-loading utilities (preprocessing, PyTorch Dataset/DataLoader wrappers).
- `layers.py` — Core Transformer building blocks: attention, multi-head attention, positional encodings, feed-forward layers, normalization, and composite encoder/decoder blocks.
- `model.py` — High-level model assembly that composes layers into an encoder/decoder or autoregressive model as needed by experiments.
- `losses.py` — Loss function(s) used for training (task-specific wrappers around PyTorch losses).
- `train.py` — Training loop and trainer utilities (epoch loop, logging, checkpoint save/load).
- `main.py` — CLI/entrypoint for running training, evaluation, or inference (argument parsing).
- `utils.py` — Miscellaneous helper functions (seed setting, metric calculations, checkpoint helpers, device utilities).

## Requirements
- Python 3.8+
- PyTorch (tested with 1.8+; use the version appropriate for your CUDA)
- numpy
- tqdm
Optional:
- tensorboard or wandb for logging (if referenced in train/main)
Install with pip:
pip install torch numpy tqdm

Or use a requirements file if you add one:
pip install -r requirements.txt

## Quickstart

1. Create and activate a virtual environment (recommended):
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies:
   pip install torch numpy tqdm

3. Explore CLI options:
   python main.py --help
   python train.py --help

4. Run training (example):
   python main.py --mode train --config path/to/config.yaml
   (If your project uses simple args instead of a config file, use the flags shown in `--help`.)

5. Run evaluation / inference:
   python main.py --mode eval --checkpoint path/to/checkpoint.pt
   python main.py --mode infer --checkpoint path/to/checkpoint.pt --input "your input text"

Notes: Exact CLI flags depend on how `main.py` and `train.py` parse arguments. Use the `--help` output to see available options.

## Typical workflow
- Prepare dataset or point `dataset.py` to your data / preprocessing.
- Configure model hyperparameters (embedding dim, number of heads, layers) in the place `main.py` or `train.py` expects (CLI flags, config file, or constants).
- Train using `train.py` or `main.py` (entrypoint).
- Save checkpoints and evaluate. Use utilities from `utils.py` to resume or load checkpoints.

## Files to inspect / extend
- Start with `layers.py` to understand how attention and Transformer blocks are implemented.
- `model.py` shows how layers are assembled into a full model.
- `dataset.py` contains the dataset and batching logic; adapt it for new datasets.
- `train.py` contains the training logic and typical hooks (checkpoints, logging, validation).
- `utils.py` contains small helpers reused across files — extend as needed.

## Development tips
- Use small subsets of data and reduced model sizes for quick iteration and debugging.
- Set deterministic seeds in `utils.py` if reproducibility is required.
- Add a `requirements.txt` and a `config` example (e.g., YAML or JSON) to make experiments reproducible.
- Add unit tests for critical layers in `layers.py` (e.g., compare attention shapes, mask behavior).

## Contribution
If you want to contribute:
- Open issues for bugs or feature requests.
- Add tests for new features or refactors.
- Follow consistent coding and docstring style.

## License
Specify your preferred license (e.g., MIT). If you don't have one yet, add a `LICENSE` file at the repo root.
