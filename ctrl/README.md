# CTRL Module (Basic CTRL-style Transformer)

This directory contains a minimal, educational implementation of a CTRL-style conditional transformer used to demonstrate how to condition generation on control codes. The code is intended for learning and experimentation rather than production use.

## Contents

- `main.py` — Example script that instantiates the model, runs a short training loop on dummy data, and demonstrates conditional generation using control codes.
- `dataset.py` — Dummy dataset / dataloader utilities used for training and demonstration.
- `model.py` — High-level model wrapper; defines the `BasicCTRLModel` used by the rest of the code.
- `layers.py` — Low-level transformer components (attention, feed-forward, embeddings, etc.).
- `train.py` — Training loop and helper functions.
- `inference.py` — Auto-regressive generation utilities (sampling / generation loop).

## Quick overview

This implementation shows how you can prepend or otherwise condition a transformer on short control codes (special token IDs) to influence generation. The code uses dummy/random data in the example, so generated token IDs will be random unless you replace the dataset and tokenizer with real data.

## Requirements

- Python 3.8+
- PyTorch (1.10+ recommended)

Install basic dependencies (adjust CUDA/cpu builds as needed):

```
# Example (CPU):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# Or for CUDA (example):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

If you have a repository-level requirements file, prefer installing that:

```
python -m pip install -r requirements.txt
```

## How to run

From the repository root, run:

```
python ctrl/main.py
```

What this does:
- Instantiates `BasicCTRLModel` with example hyperparameters (vocab size, d_model, etc.).
- Runs a short training loop on a dummy dataloader provided by `dataset.py`.
- Demonstrates generation conditioned on two example control codes (`control_code_horror_id` and `control_code_comedy_id`).

Notes:
- The script uses randomly generated token IDs and dummy training data, so model outputs are not meaningful text. To get real results, integrate a tokenizer (BPE/Byte-Pair Encoding or similar), prepare a real dataset, and train for more steps.

## Files summary (what to edit)

- `dataset.py`: Replace the dummy dataloader with your real dataset loader. Ensure it yields input token IDs and appropriate attention/mask tensors.
- `model.py` / `layers.py`: The implementation contains modular components you can adapt (embedding size, positional encodings, attention masks, how control codes are concatenated/combined). If you want to use textual control codes, map them to special token IDs with your tokenizer.
- `train.py`: Customize training hyperparameters, checkpoint saving, logging, and dataset batching.
- `inference.py`: Tweak sampling strategy (greedy, top-k/top-p, temperature) and EOS handling.

## Example workflow to get real text outputs

1. Add a tokenizer (e.g., Hugging Face Tokenizers or a GPT-2 BPE vocabulary) and map control phrases to token IDs.
2. Replace the dummy dataset with preprocessed tokenized training data (input IDs, labels, attention masks).
3. Train for sufficient steps (and save checkpoints). You may want to start from pre-trained weights for practical results.
4. Use `inference.generate_text` with tokenizer.decode to convert output token IDs back to text.

## Notes & known limitations

- This is a small, pedagogical codebase. It omits many production practices: efficient batching, FP16 mixed precision, checkpointing, validation loops, scheduler management, and robust tokenization.
- The example hyperparameters in `main.py` (vocab size, layer counts, hidden sizes) are placeholders — tune them to your resources and dataset.

## Contributing

Contributions are welcome. If you plan to extend this module:
- Add a `requirements.txt` or `environment.yml` describing reproducible dependencies.
- Add a dataset preprocessing notebook or script.
- Add checkpoint saving and evaluation scripts.

## License

See the repository-level LICENSE file for license terms.

