import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state # N in the paper

        # Linear projection for expansion (W_in in the paper, combined with W_conv)
        self.in_proj = nn.Linear(d_model, d_inner * 2)

        # Causal 1D convolution
        # Equivalent to convolution + activation in Mamba (S4D paper uses `d_inner` for convolution kernel length)
        # Here, `d_inner` is the number of channels, `expand` is the factor
        self.conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=3, padding=2, groups=d_inner)

        # x_proj for B and C
        # B and C are (d_inner, d_state) each, so output is d_inner * d_state * 2
        self.x_proj = nn.Linear(d_inner, d_inner * d_state * 2) # projects to (B, C) for each d_inner
        self.dt_proj = nn.Linear(d_inner, d_inner) # projects to delta for each d_inner

        # Learnable parameters A, D
        # A is state matrix (d_inner, d_state)
        # Initialize A as in S4 paper (log space for stability)
        # A_log should be initialized such that exp(A_log) leads to stable A matrix
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

        print(f"MambaBlock initialized with d_model={d_model}, d_inner={d_inner}, d_state={d_state}")

    def _selective_scan(self, u, delta, A, B, C, D):
        # u: (batch_size, seq_len, d_inner)
        # delta: (batch_size, seq_len, d_inner)
        # A: (d_inner, d_state)
        # B: (batch_size, seq_len, d_inner, d_state)
        # C: (batch_size, seq_len, d_inner, d_state)
        # D: (d_inner)

        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize A and B (A_bar, B_bar)
        # delta is already (batch_size, seq_len, d_inner) after dt_proj, apply softplus and unsqueeze for broadcasting
        delta_t = F.softplus(delta).unsqueeze(-1) # (batch_size, seq_len, d_inner, 1)

        # A: (d_inner, d_state) -> (1, 1, d_inner, d_state)
        A = torch.exp(self.A_log).unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(-delta_t * A)

        # B is already (batch_size, seq_len, d_inner, d_state)
        # B_bar = B * (delta * A_inv) * (exp(delta A) - I)
        B_bar = B * (delta_t * A).exp().sub(1).div(A) * delta_t # Mamba's exact formula for diagonal A
        B_bar = torch.where(B_bar.isnan(), delta_t, B_bar) # Handle division by zero for small A (limit)

        # Hidden states (x) and outputs (y)
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device) # (batch_size, d_inner, d_state)
        ys = []

        u = u.transpose(1, 2) # (batch_size, d_inner, seq_len)
        B_bar = B_bar.transpose(1, 2) # (batch_size, d_inner, seq_len, d_state)
        C = C.transpose(1, 2) # (batch_size, d_inner, seq_len, d_state)
        A_bar = A_bar.transpose(1, 2) # (batch_size, d_inner, seq_len, d_state)

        for t in range(seq_len):
            # x: (batch_size, d_inner, d_state)
            # A_bar: (batch_size, d_inner, d_state) - (from A_bar[:,:,t,:] for current time step)
            # B_bar: (batch_size, d_inner, d_state) - (from B_bar[:,:,t,:] for current time step)
            # u: (batch_size, d_inner) - (from u[:,:,t] for current time step)

            # x = A_bar * x + B_bar * u
            x = A_bar[:,:,t,:] * x + B_bar[:,:,t,:] * u[:,:,t].unsqueeze(-1) # (batch_size, d_inner, d_state)

            # y = C * x + D * u
            y = (C[:,:,t,:] * x).sum(dim=-1) + D * u[:,:,t] # (batch_size, d_inner)
            ys.append(y)

        return torch.stack(ys, dim=1) # (batch_size, seq_len, d_inner)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        x_norm = self.norm(x) # (batch_size, seq_len, d_model)

        # In Mamba, the input is first expanded (d_model -> 2 * d_inner)
        # then split into two branches: conv-SSM and gate
        x_proj = self.in_proj(x_norm) # (batch_size, seq_len, 2 * d_inner)

        # Split into conv_input and gate_input
        conv_input, gate_input = x_proj.chunk(2, dim=-1) # Each (batch_size, seq_len, d_inner)

        # Causal 1D convolution (operates on the second last dim: seq_len)
        # Pytorch Conv1d expects (batch_size, channels, seq_len)
        # So, permute conv_input to (batch_size, d_inner, seq_len)
        conv_input = conv_input.transpose(1, 2) # (batch_size, d_inner, seq_len)

        # Apply causal convolution. padding='same' ensures output has same seq_len
        # For kernel_size=3 and padding=2, the effective receptive field for padding='causal'
        # means we need to pad (kernel_size - 1) on the left. Pytorch's padding is symmetric.
        # So, manually pad for causal convolution
        conv_input_padded = F.pad(conv_input, (self.conv1d.kernel_size[0] - 1, 0)) # Pad left
        conv_output = self.conv1d(conv_input_padded)[:, :, :seq_len] # (batch_size, d_inner, seq_len)

        # Apply SiLU activation
        conv_output = self.act(conv_output) # (batch_size, d_inner, seq_len)

        # Transpose back to (batch_size, seq_len, d_inner) for SSM
        x_bc = conv_output.transpose(1, 2) # (batch_size, seq_len, d_inner) - This `x_bc` acts as input to dt_proj and x_proj

        # Delta projection
        delta_proj_out = self.dt_proj(x_bc) # (batch_size, seq_len, d_inner)

        # B and C projection
        bc_proj_out = self.x_proj(x_bc) # (batch_size, seq_len, d_inner * d_state * 2)

        # Reshape B and C
        B_flat, C_flat = bc_proj_out.chunk(2, dim=-1) # Each (batch_size, seq_len, d_inner * d_state)
        B = B_flat.view(batch_size, seq_len, self.d_inner, self.d_state) # (batch_size, seq_len, d_inner, d_state)
        C = C_flat.view(batch_size, seq_len, self.d_inner, self.d_state) # (batch_size, seq_len, d_inner, d_state)

        # Apply selective scan
        ssm_output = self._selective_scan(u=x_bc, delta=delta_proj_out, A=self.A_log, B=B, C=C, D=self.D)

        # Multiply by gate_input after activation
        gate = self.act(gate_input) # (batch_size, seq_len, d_inner)
        output_gated = ssm_output * gate # (batch_size, seq_len, d_inner)

        # Output projection and residual connection
        output = self.out_proj(output_gated) # (batch_size, seq_len, d_model)

        return output + x # Residual connection
    
class GlobalSharedAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if (d_model % n_heads) != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        print(f"GlobalSharedAttention initialized with d_model={d_model}, n_heads={n_heads}")

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Apply LayerNorm before attention for pre-norm architecture
        norm_x = self.norm(x)

        # Project to Q, K, V
        Q = self.q_proj(norm_x) # (batch_size, seq_len, d_model)
        K = self.k_proj(norm_x) # (batch_size, seq_len, d_model)
        V = self.v_proj(norm_x) # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        # (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (Q * K^T / sqrt(head_dim))
        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask (if provided). For causal language modeling, this is typically an upper-triangular mask.
        # This GlobalSharedAttention implies attending to all previous tokens, so a causal mask is appropriate
        # if this layer is meant to respect causality within a sequence. For pure global attention (e.g., cross-attention),
        # a causal mask might not be used or `mask` would be an external input.
        if mask is None:
            # Create a causal mask for language modeling task
            seq_len = Q.size(-2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
        elif mask is not None:
            # If an explicit mask is passed (e.g., padding mask), apply it
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Multiply weights with Value tensor
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, head_dim)
        # -> (batch_size, n_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear projection
        # (batch_size, seq_len, n_heads, head_dim) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(output)

        # Residual connection
        return x + output