# Custom GPT (Decoder-Only Transformer) Implementation

This directory contains the core implementation files for a scratch-built, character/word-level autoregressive **Generative Pre-trained Transformer (GPT)** language model.

## ⚙️ How It Works: The Architecture Break Down

Unlike encoder-decoder setups (like BERT or standard translation T5 models), a **GPT model is decoder-only**. It is designed strictly to take a sequence of text tokens and predict the absolute next token in line.

### 1. Token & Positional Embedding Layer (`dataset.py` & `model.py`)
* Text data is ingested, mapped to integers (indices), and fed through an `nn.Embedding` layer.
* Because self-attention has no inherent sense of sequence order, we inject **Learned Positional Embeddings** to give the model structural awareness of where words or characters live inside the context window.

### 2. Multi-Head Causal Self-Attention Layer (`model.py`)
* The core brain of the model. It projects input sequences into **Queries (Q)**, **Keys (K)**, and **Values (V)** vectors.
* **Causal Masking:** A lower-triangular mask matrix is applied to the scaled dot-product attention scores. This zeroes out future tokens, ensuring that when the model predicts token $N$, it can *only* look at tokens $1$ through $N-1$.

### 3. Feed-Forward & LayerNorm Blocks
* After attention blocks extract spatial text features, the data passes through a multi-layer perceptron (Linear ➡️ GELU activation ➡️ Linear) to add non-linear representation capacity.
* **Pre-LN (Layer Normalization):** Placed right before the attention and feed-forward blocks to keep gradient distributions stable across deeper layers.

---

## 📊 Hyperparameters to Tweak inside `train.py`

When training your custom model (e.g., on a text file like Shakespeare or a small book corpus), you can adjust these core variables to fit your computer's hardware:

| Hyperparameter | Typical Default Value | Description |
| :--- | :--- | :--- |
| `block_size` (Context) | `64` to `256` | Max sequence length the model reads at one time |
| `n_embd` (Embedding Dim) | `128` to `384` | Channel depth size of internal layer vectors |
| `n_head` | `4` to `6` | Number of parallel attention heads running |
| `n_layer` | `4` to `6` | Number of stacked Transformer blocks |
| `learning_rate` | `3e-4` (AdamW) | Standard optimal learning rate for transformers |

---

## 🔮 Inference & Text Generation

To evaluate how well your network learned, the `generate.py` script takes a starting text seed, converts it into input tokens, and passes it through the model iteratively:

$$\text{Prompt} \rightarrow \text{Predict Next Token} \rightarrow \text{Append Token to Input} \rightarrow \text{Repeat}$$

Using temperature scaling during generation prevents the model from choosing the exact same words repeatedly, giving your model a natural, creative writing output style!
