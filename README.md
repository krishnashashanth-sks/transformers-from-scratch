# Transformers Zoo: Custom Autoregressive Language Models

Welcome to my repository dedicated to exploring, implementing, and training Transformer architectures from scratch in PyTorch. This project serves as a practical, modular guide to understanding self-attention, positional encodings, and autoregressive generation without relying on high-level model libraries.

## 📌 Project Overview
The introduction of the Transformer architecture revolutionized Natural Language Processing (NLP). This repository breaks down complex sequence-to-sequence and autoregressive math into clean, decoupled, and well-documented PyTorch modules so you can learn by reading and running the code.

---

## 🤖 Architectures Maintained

* **Decoder-Only Transformer (GPT from Scratch):** A generative language model designed for next-token prediction. It processes tokenized text sequences, applies causal self-attention masks to prevent information leakage from future tokens, and is implemented as a stack of attention + MLP blocks.

* **Basic CTRL-style Conditional Transformer (CTRL):** A minimal implementation that demonstrates conditioning generation on control codes (special token IDs) so the model can generate text in different styles or formats depending on a prepended control token.

* **IQ-Former:** An experimental transformer variant contained in the `iq-former` module. This folder hosts an implementation exploring alternative attention or architectural tweaks (see `iq-former/README.md` for details).

* **TReCVit:** A transformer variant in the `trecvit` folder focused on [experimental task / multimodal ideas]. Check `trecvit/README.md` for specifics and usage examples.

---

(See the subdirectories `gpt/`, `ctrl/`, `iq-former/`, and `trecvit/` for module-specific READMEs, usage examples, and implementation details.)
