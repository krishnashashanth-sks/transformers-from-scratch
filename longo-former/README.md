# Longformer: Long Document Transformer

A clean-room implementation of the **Longformer** model architecture from scratch using PyTorch, designed for efficient processing of long documents with linear complexity through local and global attention mechanisms.

## Overview

Longformer is an advanced Transformer variant that addresses the quadratic complexity of standard transformers when processing long sequences. This implementation features:

- **Local Attention**: Sliding window attention for processing local context efficiently
- **Global Attention**: Selective global attention for important tokens spanning the entire document
- **Configurable Attention Windows**: Flexible window sizes for different use cases
- **Modular Components**: Clean separation of concerns with distinct layers and model classes

## Architecture

### Key Components

#### 1. **Longformer Configuration** (`config.py`)
Centralizes hyperparameter management:

```python
LongformerConfig(
    vocab_size=30522,              # Vocabulary size (BERT-like)
    hidden_size=768,               # Hidden dimension
    num_hidden_layers=12,          # Number of encoder layers
    num_attention_heads=12,        # Number of attention heads
    intermediate_size=3072,        # FFN intermediate dimension
    max_position_embeddings=4096,  # Maximum sequence length
    attention_window=512,          # Sliding window size (Longformer-specific)
    num_labels=3,                  # Number of classification labels
)
```

#### 2. **Layers** (`layers.py`)

**LongformerSelfAttention**
- Implements local sliding window attention with efficient computation
- Supports global attention for specific tokens
- Handles both local and global attention patterns:
  - **Local**: Limited to a sliding window around each position
  - **Global**: Unrestricted attention for designated tokens
- Robust numerical stability with NaN/Inf protection

**LongformerEncoderLayer**
- Combines self-attention with feed-forward networks
- Residual connections for better gradient flow
- Layer normalization for training stability
- Full encoder block following the Transformer pattern

#### 3. **Model** (`model.py`)

**LongformerModel**
- Stacks multiple encoder layers
- Embedding layers: word, positional, and token type embeddings
- Classification head for downstream tasks
- Weight initialization following best practices

### Attention Mechanism

#### Local Attention (Sliding Window)
```
For each query position q:
  Attention scope = [q - window_size/2, q + window_size/2]
  - Efficiently computes local context
  - Linear complexity: O(n * window_size)
```

#### Global Attention
```
Selected tokens can attend to ALL positions:
  - Typically: [CLS] token, document boundaries
  - Bridges local windows for global information flow
  - Enables document-level understanding
```

## File Structure

```
longo-former/
├── config.py          # Configuration management
├── layers.py          # LongformerSelfAttention & LongformerEncoderLayer
├── model.py           # LongformerModel architecture
├── tokenizer.py       # Token preprocessing & encoding
├── dataset.py         # Data loading and preprocessing
├── train.py           # Training loop and optimization
├── main.py            # End-to-end training pipeline
├── inference.py       # Model inference utilities
└── README.md          # This file
```

## Usage

### 1. Configuration Setup

```python
from config import LongformerConfig

config = LongformerConfig(
    vocab_size=30522,
    hidden_size=768,
    attention_window=512
)
```

### 2. Model Initialization

```python
from model import LongformerModel

model = LongformerModel(config)
```

### 3. Forward Pass

```python
import torch

batch_size, seq_len = 4, 1024
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
attention_mask = torch.ones((batch_size, seq_len))

# Optional: Mark specific tokens for global attention (e.g., [CLS])
global_attention_mask = torch.zeros((batch_size, seq_len))
global_attention_mask[:, 0] = 1  # First token gets global attention

logits = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    global_attention_mask=global_attention_mask
)
# Output shape: (batch_size, num_labels)
```

### 4. Training

```python
python main.py --config config.yaml
```

## Key Features

### Numerical Stability
- **NaN/Inf Handling**: Extensive checks and sanitization throughout computation
- **Attention Score Clamping**: Prevents overflow/underflow in softmax
- **Robust Padding Masking**: Proper handling of variable-length sequences

### Efficiency
- **Linear Attention Complexity**: O(n) instead of O(n²) for long documents
- **Configurable Window Sizes**: Trade-off between locality and compute
- **Sparse Attention Pattern**: Only computes relevant attention scores

### Flexibility
- **Hybrid Attention**: Combine local and global patterns dynamically
- **Adaptive Window Sizes**: Configure for different document lengths
- **Multi-head Architecture**: 12 attention heads for diverse representations

## Mathematical Details

### Sliding Window Attention

For sequence of length `L` and window size `W`:
- Each query attends to positions in range `[i - W/2, i + W/2]`
- Computational complexity: **O(L × W)** instead of **O(L²)**

### Global Attention Integration

```
Attention(query_q, keys, values) = 
  - Sliding window attention if q is local
  - Full attention if q is global (attends to all positions)
```

### Feed-Forward Network

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
Gate(x) = GELU(xW₁ + b₁)W₂ + b₂  # In this implementation
```

## Performance Characteristics

| Metric | Standard Transformer | Longformer |
|--------|---------------------|-----------|
| Time Complexity | O(n²) | O(n × w) |
| Space Complexity | O(n²) | O(n × w) |
| Max Sequence Length | ~512 | 4096+ |
| Training Speed | Baseline | 4-8× faster |

*where n = sequence length, w = window size*

## Hyperparameter Tuning

### For Long Documents (>2K tokens)
```python
attention_window=512    # Larger window for broader context
hidden_size=1024        # Increased capacity
num_attention_heads=16  # More diverse representations
```

### For Classification Tasks
```python
hidden_size=768
num_hidden_layers=12
intermediate_size=3072
num_labels=3            # Task-specific
```

## Implementation Highlights

### Numerical Robustness
- Sanitization after matmul and softmax operations
- Handling of -inf attention scores
- Prevention of NaN propagation through careful masking

### Memory Efficiency
- Sliding window attention reduces memory from O(n²) to O(n·w)
- Efficient tensor operations with proper reuse

### Code Quality
- Modular design separating attention, encoding, and model logic
- Comprehensive docstrings and comments
- Type hints for better code clarity

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Tokenizers (for preprocessing)

## Future Enhancements

- [ ] Implement DistilLongformer (knowledge distillation)
- [ ] Add pre-training objectives (MLM, NSP)
- [ ] Support for multiple document languages
- [ ] Quantization for deployment
- [ ] Distributed training support

## References

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Beltagy et al.
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.

## Author

Krishna Shashanth K S  
GitHub: [@krishnashashanth-sks](https://github.com/krishnashashanth-sks)

## License

MIT License - See LICENSE file for details

---

**Built with ❤️ for efficient long document processing**
