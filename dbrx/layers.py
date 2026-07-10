import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    """Combines token and positional embeddings."""
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        # Create positional indices
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Get token and positional embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings (usually by summing)
        embeddings = token_embeds + position_embeds
        return embeddings
    
class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # (batch_size, num_heads, seq_len_q, head_dim) @ (batch_size, num_heads, head_dim, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Mask shape from Dataloader is (batch_size, seq_len_k)
            # scores shape is (batch_size, num_heads, seq_len_q, seq_len_k)
            # Expand mask to (batch_size, 1, 1, seq_len_k) for correct broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2) # -> (batch_size, 1, 1, seq_len_k)

            # Explicitly expand mask to scores shape to avoid broadcasting issues with masked_fill
            # This makes the mask (batch_size, num_heads, seq_len_q, seq_len_k) before applying.
            expanded_mask_for_fill = mask_expanded.expand_as(scores)

            scores = scores.masked_fill(expanded_mask_for_fill == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        context_layer = torch.matmul(attention_weights, v)

        # Concatenate heads and project back
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context_layer)

        return output
    
class FeedForwardNetwork(nn.Module):
    """Standard Feed-Forward Network component."""
    def __init__(self, embed_dim: int, inner_dim: int, activation=nn.GELU, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, inner_dim)
        self.activation = activation()
        self.linear2 = nn.Linear(inner_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class NormalizationLayer(nn.Module):
    """Normalization layer (e.g., Layer Normalization)."""
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class ExpertNetwork(nn.Module):
    """A simple feed-forward network that acts as an expert in the MoE layer."""
    def __init__(self, embed_dim: int, hidden_dim: int, activation=nn.GELU, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GatingNetwork(nn.Module):
    """A network to determine which experts to route tokens to."""
    def __init__(self, embed_dim: int, num_experts: int):
        super().__init__()
        # Linear layer to project input to scores for each expert
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, seq_len, embed_dim)
        # The gate should output scores for each expert for each token
        # Output shape: (batch_size, seq_len, num_experts)
        return self.gate(x)


class MoELayer(nn.Module):
    """Mixture-of-Experts (MoE) layer with top-k gating and optional load balancing."""
    def __init__(self, embed_dim: int, hidden_dim: int, num_experts: int, top_k: int = 2,
                 activation=nn.GELU, dropout: float = 0.1, enable_load_balancing: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.enable_load_balancing = enable_load_balancing

        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts})")

        self.gating_network = GatingNetwork(embed_dim, num_experts)
        self.experts = nn.ModuleList([
            ExpertNetwork(embed_dim, hidden_dim, activation, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        flat_x = x.view(-1, embed_dim) # Flatten for gating and expert processing

        # Get expert scores from the gating network
        # shape: (batch_size * seq_len, num_experts)
        gate_scores = self.gating_network(flat_x)
        gate_weights = F.softmax(gate_scores, dim=-1) # Probabilities for each expert

        # Select top-k experts
        # top_k_weights: (batch_size * seq_len, top_k), top_k_indices: (batch_size * seq_len, top_k)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Initialize output tensor
        output = torch.zeros_like(flat_x)

        # Create a tensor to store expert load (for load balancing loss if enabled)
        expert_load = torch.zeros(self.num_experts, device=x.device)
        # Create a mask to scatter input tokens to experts
        # (batch_size * seq_len, num_experts)
        expert_assignment_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).bool()

        # Iterate through experts and process tokens assigned to them
        for i, expert in enumerate(self.experts):
            # Get mask for tokens routed to this expert
            # This expert is chosen if its index is in top_k_indices for a token
            expert_mask = (top_k_indices == i).any(dim=-1)

            if expert_mask.any():
                # Get indices of tokens routed to this expert
                routed_indices = torch.nonzero(expert_mask, as_tuple=True)[0]

                # Apply expert to routed tokens
                expert_output = expert(flat_x[routed_indices])

                # Get the weights for this expert for the routed tokens
                # (num_routed_tokens, top_k) -> (num_routed_tokens,)
                expert_weights_for_tokens = top_k_weights[routed_indices]
                expert_chosen_column = (top_k_indices[routed_indices] == i).nonzero(as_tuple=True)[1]
                specific_expert_weights = expert_weights_for_tokens.gather(1, expert_chosen_column.unsqueeze(1)).squeeze(1)

                # Accumulate weighted expert output
                output[routed_indices] += expert_output * specific_expert_weights.unsqueeze(-1)

                if self.enable_load_balancing:
                    expert_load[i] = expert_mask.sum().float()

        # Reshape output back to (batch_size, seq_len, embed_dim)
        output = output.view(batch_size, seq_len, embed_dim)

        if self.enable_load_balancing:
            # Calculate load balancing loss component (simplified version)
            # This encourages experts to be equally utilized
            # Load balancing loss = sum(gate_weights) * sum(expert_load_indicator)
            # where expert_load_indicator is 1 if expert was chosen for a token, 0 otherwise.
            # A more sophisticated loss would consider 'importance_loss' and 'load_loss' as in original MoE papers.
            # For simplicity, we can use a proxy like the variance of expert loads or a combination with gate_weights.

            # Simplified load balancing loss term: Encourage gate weights to be uniform for chosen experts.
            # This is a very basic proxy and actual MoE implementations use a more complex loss.
            # Here, we'll just track expert load and leave the actual loss calculation for the training loop.
            # A more direct approach for a callable loss is to return `gate_weights` and `expert_assignment_mask`
            # and calculate the auxiliary loss outside.

            # As per instructions, we just need to define the component.
            # The load balancing loss will be calculated in the training step.
            pass # No direct loss calculation here, just ensure routing is ready.

        return output

class TransformerBlock(nn.Module):
    """A single Transformer block with pre-normalization, multi-head self-attention, and MoE layer."""
    def __init__(self, embed_dim: int, num_heads: int, moe_hidden_dim: int, num_experts: int, top_k: int,
                 activation=nn.GELU, dropout: float = 0.1):
        super().__init__()

        # Pre-normalization for attention
        self.norm1 = NormalizationLayer(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        # Pre-normalization for MoE
        self.norm2 = NormalizationLayer(embed_dim)
        self.moe = MoELayer(embed_dim, moe_hidden_dim, num_experts, top_k, activation, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention block with pre-normalization and residual connection
        # Apply normalization before attention
        norm_x = self.norm1(x)
        attn_output = self.attn(norm_x, mask=mask)
        x = x + self.dropout1(attn_output) # Residual connection

        # MoE block with pre-normalization and residual connection
        # Apply normalization before MoE
        norm_x = self.norm2(x)
        moe_output = self.moe(norm_x)
        x = x + self.dropout2(moe_output) # Residual connection

        return x
