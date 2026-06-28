from layers import *
import torch.nn as nn

class BERT(nn.Module):
  def __init__(self,vocab_size,embed_dim,max_seq_len,num_segments,num_layers,num_heads,ff_dim,dropout_rate):
    super().__init__()
    self.embedding=BERTEmbedding(vocab_size,embed_dim,max_seq_len,num_segments,dropout_rate)
    self.encoder_blocks=nn.ModuleList([
        EncoderBlock(embed_dim,num_heads,ff_dim,dropout_rate)
        for _ in range(num_layers)
    ])
  def forward(self,token_ids,segment_ids,mask=None):
    x=self.embedding(token_ids,segment_ids)
    for encoder_block in self.encoder_blocks:
      x=encoder_block(x,mask)
    return x
  
class BERTForPretraining(BERT):
  def __init__(self,vocab_size,embed_dim,max_seq_len,num_segments,num_layers,num_heads,ff_dim,dropout_rate):
    super().__init__(vocab_size,embed_dim,max_seq_len,num_segments,num_layers,num_heads,ff_dim,dropout_rate)
    self.mlm_head=MaskedLanguageModelHead(embed_dim,vocab_size)
    self.nsp_head=NextSentencePredictionHead(embed_dim)
  def forward(self,token_ids,segment_ids,mask=None):
    encoder_output=super().forward(token_ids,segment_ids,mask)
    mlm_prediction_scores=self.mlm_head(encoder_output)
    cls_output=encoder_output[:,0,:]
    nsp_prediction_scores=self.nsp_head(cls_output)
    return mlm_prediction_scores,nsp_prediction_scores

class BERTWithMLM(BERT):
  def __init__(self,vocab_size,embed_dim,max_seq_len,num_segments,num_layers,num_heads,ff_dim,dropout_rate):
    super().__init__(vocab_size,embed_dim,max_seq_len,num_segments,num_layers,num_heads,ff_dim,dropout_rate)
    self.mlm_head=MaskedLanguageModelHead(embed_dim,vocab_size)
  def forward(self,token_ids,segment_ids,mask=None):
    encoder_output=super().forward(token_ids,segment_ids,mask)
    return self.mlm_head(encoder_output)