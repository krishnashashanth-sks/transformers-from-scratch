import torch
import torch.nn as nn
from layers import SambaBlock

class SambaModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int,
                 d_state: int, num_heads: int, window_size: int,
                 d_inner_mamba: int = None, conv_kernel_size: int = 4,
                 d_inner_swiglu: int = None, bias: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # 2a. Initialize an embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2b & 2c. Create a ModuleList to hold multiple SambaBlock instances
        self.layers = nn.ModuleList([
            SambaBlock(
                d_model=d_model,
                d_state=d_state,
                num_heads=num_heads,
                window_size=window_size,
                d_inner_mamba=d_inner_mamba,
                conv_kernel_size=conv_kernel_size,
                d_inner_swiglu=d_inner_swiglu,
                bias=bias
            )
            for _ in range(num_layers)
        ])

        # 2d. Initialize a final linear layer for output predictions
        self.out_projection = nn.Linear(d_model, vocab_size)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)

        # 3a. Pass input token IDs through the embedding layer
        x = self.embedding(input_ids) # (batch, seq_len, d_model)

        # 3b. Iterate through the SambaBlocks
        for layer in self.layers:
            x = layer(x)

        # 3c. Apply the final linear layer
        output = self.out_projection(x) # (batch, seq_len, vocab_size)

        # 3d. Return the final output tensor
        return output