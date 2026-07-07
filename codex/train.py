import torch.nn as nn
from utils import create_attention_mask
import torch

# Inside your training loop, ensure gradients are clipped to prevent explosion
def train_step(model, optimizer, input_ids, labels, pad_token_id, device):
    model.train()
    optimizer.zero_grad()

    mask = create_attention_mask(input_ids, pad_token_id, device)
    outputs = model(input_ids, mask=mask)

    # CrossEntropyLoss ignores pad_token_id automatically if set
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))

    loss.backward()

    # GRADIENT CLIPPING: Crucial for Transformer stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()
