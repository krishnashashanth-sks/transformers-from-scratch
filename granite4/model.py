import torch.nn as nn
from layers import TransformerBlock
import torch

class Granite4_0Model(nn.Module):
  def __init__(self,vocab_size,max_seq_len,hidden_size,num_heads,num_layers,dropout_rate,intermediate_size):
    super().__init__()
    self.token_embeddings=nn.Embedding(vocab_size,hidden_size)
    self.position_embeddings=nn.Embedding(max_seq_len,hidden_size)
    self.dropout=nn.Dropout(dropout_rate)
    self.transformer_blocks=nn.ModuleList([
        TransformerBlock(hidden_size,num_heads,intermediate_size,dropout_rate)
        for _ in range(num_layers)
    ])
    self.final_norm=nn.LayerNorm(hidden_size)
    self.lm_head=nn.Linear(hidden_size,vocab_size,bias=False)
    self.apply(self._init_weights)
    # Tie weights after initialization
    self.lm_head.weight = self.token_embeddings.weight

  def _init_weights(self,m):
    if isinstance(m,nn.Linear):
      nn.init.normal_(m.weight,mean=0.0,std=0.02)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Embedding):
      nn.init.normal_(m.weight,mean=0.0,std=.02)
    elif isinstance(m,nn.LayerNorm):
      nn.init.ones_(m.weight)
      nn.init.zeros_(m.bias)

  def forward(self, input_ids, attention_mask=None): # Fixed 'inputs_ids' to 'input_ids'
    batch_size,seq_len=input_ids.shape
    token_embeds=self.token_embeddings(input_ids)
    position_ids=torch.arange(0,seq_len,dtype=torch.long,device=input_ids.device)
    position_embeds=self.position_embeddings(position_ids)
    x=self.dropout(token_embeds+position_embeds)

    # Generate causal mask for decoder-only transformers
    # The mask should be (seq_len, seq_len) and applied directly to attn_mask argument
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)

    for block in self.transformer_blocks:
      x=block(x,attention_mask=attention_mask, causal_mask=causal_mask) # Pass the causal_mask
    return self.lm_head(self.final_norm(x))