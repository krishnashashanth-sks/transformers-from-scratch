import torch.nn as nn
from layers import EmbeddingLayer,TransformerBlock,ContextEncoder,Hypernetwork

class HOPE_Architecture(nn.Module):
  def __init__(self,vocab_size,d_model,num_heads,d_ff,num_layers,max_len=512,dropout_rate=0.1,context_dim=256):
    super().__init__()
    self.d_model=d_model
    self.num_heads=num_heads
    self.d_ff=d_ff
    self.num_layers=num_layers
    self.context_dim=context_dim
    self.embedding_layer=EmbeddingLayer(vocab_size,d_model,max_len)
    self.transformer_blocks=nn.ModuleList([
        TransformerBlock(d_model,num_heads,d_ff,dropout_rate)for _ in range(num_layers)
    ])
    self.output_layer=nn.Linear(d_model,vocab_size)
    self.context_encoder=ContextEncoder(d_model,num_heads,context_dim)
    self.hypernetworks=nn.ModuleList([
        Hypernetwork(context_dim,d_model,num_heads,d_ff)
        for _ in range(num_layers)
    ])
  def forward(self,input_ids):
    x=self.embedding_layer(input_ids)
    current_embeddings_for_context=x.clone()
    all_fast_weight_generations=[]
    for i,block in enumerate(self.transformer_blocks):
      context_vector=self.context_encoder(current_embeddings_for_context,x)
      hypernet=self.hypernetworks[i]
      fast_weights_for_block=hypernet(context_vector)
      all_fast_weight_generations.append(fast_weights_for_block)
      x=block(x,fast_weights_attn_q=fast_weights_for_block.get('attn_q'),
                fast_weights_attn_k=fast_weights_for_block.get('attn_k'),
                fast_weights_attn_v=fast_weights_for_block.get('attn_v'),
                fast_bias_ffn1=fast_weights_for_block.get('ffn_b1'),
                fast_scale_ffn1=fast_weights_for_block.get('ffn_s1'),
                fast_bias_ffn2=fast_weights_for_block.get('ffn_b2'),
                fast_scale_ffn2=fast_weights_for_block.get('ffn_s2'))
    logits=self.output_layer(x)
    return logits,all_fast_weight_generations