import torch.nn as nn
from layers import MoETransformerBlock
import torch
from utils import _generate_causal_mask

class CustomGenerativeTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1, max_len: int = 512, num_segments: int = 4, num_experts: int = 8, top_k=2):
        super(CustomGenerativeTransformer, self).__init__()

        self.d_model = d_model
        self.max_len = max_len # Store max_len for mask generation
        self.vocab_size = vocab_size # Store vocab_size for generate method

        # 1. Input embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(num_segments, d_model) # e.g., 0: prompt, 1: instruction, 2: example, 3: response

        # 2. Transformer Encoder layers (using MoETransformerBlock)
        moe_encoder_layer = MoETransformerBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        # nn.TransformerEncoder can take a custom encoder_layer
        self.transformer_encoder = nn.TransformerEncoder(moe_encoder_layer, num_layers=num_encoder_layers)

        # 3. Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Layer normalization for embeddings
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, segment_ids: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None):
        # src: (batch_size, sequence_length) - input token IDs
        # segment_ids: (batch_size, sequence_length) - segment IDs for structured prompts
        # src_key_padding_mask: (batch_size, sequence_length) - mask for padding tokens

        seq_len = src.shape[1]
        device = src.device

        # Generate positional embeddings
        # Unsqueeze(0) makes it (1, seq_len) to broadcast with src (batch_size, seq_len)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        positional_embeddings = self.positional_embedding(positions)

        # Generate token embeddings
        token_embeddings = self.token_embedding(src)

        # Combine token and positional embeddings
        embeddings = token_embeddings + positional_embeddings

        # Add segment embeddings if provided
        if segment_ids is not None:
            segment_embeddings = self.segment_embedding(segment_ids)
            embeddings = embeddings + segment_embeddings

        # Apply layer normalization
        embeddings = self.norm(embeddings)

        # Generate causal mask if not provided. This is crucial for auto-regressive generation.
        causal_mask = _generate_causal_mask(seq_len, device)

        # Pass through Transformer Encoder
        # The 'mask' argument of TransformerEncoder will be passed as 'src_mask' to MoETransformerBlock.
        transformer_output = self.transformer_encoder(
            embeddings,
            mask=causal_mask,  # Apply the generated causal mask here
            src_key_padding_mask=src_key_padding_mask # Padding mask if provided
        )

        # Project to vocabulary size to get logits for next token prediction
        output = self.output_layer(transformer_output)

        return output

    def generate(self, initial_token_ids: torch.Tensor, segment_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int) -> torch.Tensor:
        self.eval() # Set model to evaluation mode
        device = initial_token_ids.device

        # Ensure initial_token_ids and segment_ids have batch_size=1
        if initial_token_ids.dim() == 1: # Convert (seq_len) to (1, seq_len)
            initial_token_ids = initial_token_ids.unsqueeze(0)
        if segment_ids.dim() == 1:
            segment_ids = segment_ids.unsqueeze(0)

        generated_sequence = initial_token_ids
        generated_segment_ids = segment_ids

        # Extract the segment ID for newly generated tokens. For simplicity, use the last provided segment ID.
        # In more complex scenarios, this might depend on the generation context (e.g., always 'response' segment).
        last_segment_id = segment_ids[0, -1].item() # Assuming batch_size is 1 for generation

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Check if generated sequence length exceeds max_len of positional embedding
                if generated_sequence.shape[1] >= self.max_len:
                    print("Warning: Reached maximum sequence length during generation. Stopping.")
                    break

                # Get model predictions for the current sequence
                # This forward pass uses the causal mask internally
                outputs = self.forward(generated_sequence, segment_ids=generated_segment_ids)

                # Extract logits for the last token in the sequence
                next_token_logits = outputs[:, -1, :]

                # Predict the next token (greedy approach: argmax)
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) # (1, 1)

                # Append the predicted token to the sequence
                generated_sequence = torch.cat([generated_sequence, next_token_id], dim=-1)

                # Append the corresponding segment ID (using the last one from initial input)
                generated_segment_ids = torch.cat([
                    generated_segment_ids,
                    torch.tensor([[last_segment_id]], dtype=torch.long, device=device)
                ], dim=-1)

                # Check for end-of-sequence token
                if next_token_id.item() == eos_token_id:
                    break

        self.train() # Set model back to training mode
        return generated_sequence.squeeze(0) # Remove batch dimension for output
