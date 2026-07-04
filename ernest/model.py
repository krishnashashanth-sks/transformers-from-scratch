import torch.nn as nn
import torch
from layers import TransformerEncoder

class ERNESTAdvanced(nn.Module):
  def __init__(self,num_entities,num_relations,num_types,embedding_dim,
               text_vocab_size=None,text_encoder_num_heads=4,text_encoder_num_layers=2,
               text_encoder_d_ff_ratio=4,text_max_seq_len=100, text_encoder_dropout=0.1): # Added text_encoder_dropout
    super(ERNESTAdvanced,self).__init__()
    self.embedding_dim=embedding_dim
    self.text_vocab_size=text_vocab_size
    self.entity_embeddings=nn.Embedding(num_entities,embedding_dim)
    self.relation_embeddings=nn.Embedding(num_relations,embedding_dim)
    nn.init.xavier_uniform_(self.entity_embeddings.weight)
    nn.init.xavier_uniform_(self.relation_embeddings.weight)
    self.type_embeddings=nn.Embedding(num_types,embedding_dim)
    nn.init.xavier_uniform_(self.type_embeddings.weight)
    if text_vocab_size:
      d_ff_text=embedding_dim*text_encoder_d_ff_ratio
      self.text_encoder=TransformerEncoder(text_vocab_size,
                                           embedding_dim,
                                           text_encoder_num_heads,d_ff_text,
                                           text_encoder_num_layers,
                                           dropout=text_encoder_dropout, # Pass dropout explicitly
                                           max_seq_len=text_max_seq_len)
    else:
      self.text_encoder=None
    self.entity_proj=nn.Linear(embedding_dim,embedding_dim)
    self.relation_proj=nn.Linear(embedding_dim,embedding_dim)
    self.type_proj=nn.Linear(embedding_dim,embedding_dim)
    nn.init.xavier_uniform_(self.entity_proj.weight)
    nn.init.xavier_uniform_(self.relation_proj.weight)
    nn.init.xavier_uniform_(self.type_proj.weight)
    self.head_combiner=nn.Sequential(
        nn.Linear(embedding_dim*3,embedding_dim*2),
        nn.ReLU(),
        nn.Linear(embedding_dim*2,embedding_dim),
        nn.LayerNorm(embedding_dim)
    )
    self.tail_combiner=nn.Sequential(
        nn.Linear(embedding_dim*3,embedding_dim*2),
        nn.ReLU(),
        nn.Linear(embedding_dim*2,embedding_dim),
        nn.LayerNorm(embedding_dim)
    )
    self.relation_combiner=nn.Sequential(
        nn.Linear(embedding_dim*2,embedding_dim*2),
        nn.ReLU(),
        nn.Linear(embedding_dim*2,embedding_dim),
        nn.LayerNorm(embedding_dim)
    )
    def init_weights(m):
      if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)
    self.head_combiner.apply(init_weights)
    self.tail_combiner.apply(init_weights)
    self.relation_combiner.apply(init_weights)
  def _encode_text_features(self,text_indices,batch_size,device,max_seq_len):
    if text_indices is None or self.text_encoder is None:
      return torch.zeros(batch_size,self.embedding_dim,device=device)
    current_seq_len=text_indices.shape[1]
    if current_seq_len >max_seq_len:
      text_indices=text_indices[:,:max_seq_len]
    elif current_seq_len <max_seq_len:
      # Pad with zeros to max_seq_len
      padding = torch.zeros(batch_size, max_seq_len - current_seq_len, dtype=torch.long, device=device)
      text_indices=torch.cat([text_indices,padding],dim=1)

    # The src_mask needs to be (batch_size, 1, seq_len) for the TransformerEncoder's forward method's MultiHeadAttention
    src_mask_simplified=(text_indices!=0).unsqueeze(1) # (batch_size, 1, seq_len)
    encoded_text_sequence=self.text_encoder(text_indices,src_mask=src_mask_simplified)
    non_padding_mask=(text_indices!=0).float()
    sum_embeddings=(encoded_text_sequence*non_padding_mask.unsqueeze(-1)).sum(dim=1)
    num_tokens=non_padding_mask.sum(dim=1).clamp(min=1e-9)
    pooled_embedding=sum_embeddings/num_tokens.unsqueeze(-1)
    return pooled_embedding
  def forward(self,heads,relations,tails,head_types=None,tail_types=None,
              head_text_indices=None,tail_text_indices=None,relation_text_indices=None):
    batch_size=heads.shape[0]
    device=heads.device
    h_emb_base=self.entity_embeddings(heads)
    r_emb_base=self.relation_embeddings(relations)
    t_emb_base=self.entity_embeddings(tails)
    h_emb_proj=self.entity_proj(h_emb_base)
    r_emb_proj=self.relation_proj(r_emb_base)
    t_emb_proj=self.entity_proj(t_emb_base)
    h_type_emb_proj=self.type_proj(self.type_embeddings(head_types)) if head_types is not None\
    else torch.zeros(batch_size,self.embedding_dim,device=device)
    t_type_emb_proj=self.type_proj(self.type_embeddings(tail_types)) if tail_types is not None\
    else torch.zeros(batch_size,self.embedding_dim,device=device)

    max_seq_len=self.text_encoder.positional_encoding.pe.shape[0] if self.text_encoder  else 0 # Corrected typo positional_encoding

    # Assign the results of _encode_text_features to variables
    h_text_emb = self._encode_text_features(head_text_indices,batch_size,device,max_seq_len) # Corrected typo _encode_text_features
    t_text_emb = self._encode_text_features(tail_text_indices,batch_size,device,max_seq_len) # Corrected typo _encode_text_features
    r_text_emb = self._encode_text_features(relation_text_indices,batch_size,device,max_seq_len) # Corrected typo _encode_text_features

    combined_h_features=torch.cat([h_emb_proj,h_type_emb_proj,h_text_emb],dim=1)
    h_emb_final=self.head_combiner(combined_h_features)
    combined_t_features=torch.cat([t_emb_proj,t_type_emb_proj,t_text_emb],dim=1)
    t_emb_final=self.tail_combiner(combined_t_features)

    # For relations, combine relation embedding and its text embedding
    combined_r_features = torch.cat([r_emb_proj, r_text_emb], dim=1)
    r_emb_final = self.relation_combiner(combined_r_features)

    # Corrected scoring function
    scores=torch.sum(h_emb_final * r_emb_final * t_emb_final,dim=1)
    return scores
  def loss_function(self,positive_scores,negative_scores,margin=1.0):
    loss=torch.relu(margin+negative_scores-positive_scores).mean()
    return loss
