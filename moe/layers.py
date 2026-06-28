import torch.nn as nn
import torch
import torch.nn.functional as F

class Router(nn.Module):
  def __init__(self, d_model: int, num_experts: int):
    super(Router, self).__init__()
    self.linear = nn.Linear(d_model, num_experts)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(self.linear(x), dim=-1)

class Expert(nn.Module):
  def __init__(self, d_model, dim_feedforward: int):
    super(Expert, self).__init__()
    self.ffn = nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, d_model)
    )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.ffn(x)

class MoELayer(nn.Module):
  def __init__(self, d_model: int, num_experts: int, dim_feedforward: int, top_k: int = 2):
    super(MoELayer, self).__init__()
    self.d_model = d_model
    self.num_experts = num_experts
    self.top_k = top_k
    self.router = Router(d_model, num_experts)
    self.experts = nn.ModuleList([Expert(d_model, dim_feedforward) for _ in range(num_experts)])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    device = x.device

    flat_x = x.view(-1, self.d_model)
    router_weights = self.router(flat_x)

    top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    final_output = torch.zeros_like(flat_x, device=device)

    for i in range(self.top_k):
      expert_idx = top_k_indices[:, i]
      expert_weight = top_k_weights[:, i]
      for expert_id, expert_module in enumerate(self.experts):
        mask = (expert_idx == expert_id)
        if mask.any():
          router_tokens = flat_x[mask]
          expert_output = expert_module(router_tokens)
          weighted_expert_output = expert_output * expert_weight[mask].unsqueeze(1)
          final_output[mask] += weighted_expert_output

    return final_output.view(original_shape)

class MoETransformerBlock(nn.Module):
  def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_experts: int, top_k: int = 2, dropout: float = 0.1):
    super(MoETransformerBlock, self).__init__()
    self.d_model = d_model
    self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
    self.moe_layer = MoELayer(d_model, num_experts, dim_feedforward, top_k)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None, is_causal: bool = False) -> torch.Tensor:
    attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal)
    src = src + self.dropout1(self.norm1(attn_output))
    moe_output = self.moe_layer(src)
    src = src + self.dropout2(moe_output)
    src = self.norm2(src)
    return src

