import torch.nn as nn
import torch
import math

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_segments, dropout_rate):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.segment_embeddings = nn.Embedding(num_segments, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids, segment_ids):
        seq_len = token_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        token_embed = self.token_embeddings(token_ids)
        position_embed = self.position_embeddings(position_ids)
        segment_embed = self.segment_embeddings(segment_ids)

        embeddings = token_embed + position_embed + segment_embed
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadSelfAttention(nn.Module):
  def __init__(self,embed_dim,num_heads,dropout_rate):
    super().__init__()
    self.embed_dim=embed_dim
    self.num_heads=num_heads
    self.head_dim=embed_dim//num_heads
    assert embed_dim%num_heads==0,"embed_dim must be divisble by num_heads"
    self.query=nn.Linear(embed_dim,embed_dim)
    self.key=nn.Linear(embed_dim,embed_dim)
    self.value=nn.Linear(embed_dim,embed_dim)
    self.fc_out=nn.Linear(embed_dim,embed_dim)
    self.dropout=nn.Dropout(dropout_rate)
  def forward(self,query,key,value,mask=None):
    batch_size=query.size(0)
    Q=self.query(query).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
    K=self.key(key).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
    V=self.value(value).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
    energy=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)
    if mask is not None:
      energy=energy.masked_fill(mask==0,float('1e20'))
    attention=torch.softmax(energy,dim=-1)
    x=torch.matmul(self.dropout(attention),V)
    x=x.transpose(1,2).contiguous().view(batch_size,-1,self.embed_dim)
    return self.fc_out(x)
  
class PositionwiseFeedForward(nn.Module):
  def __init__(self,embed_dim,ff_dim,dropout_rate):
    super().__init__()
    self.fc1=nn.Linear(embed_dim,ff_dim)
    self.fc2=nn.Linear(ff_dim,embed_dim)
    self.gelu=nn.GELU()
    self.dropout=nn.Dropout(dropout_rate)
  def forward(self,x):
    return self.fc2(self.dropout(self.gelu(self.fc1(x))))
  
class EncoderBlock(nn.Module):
  def __init__(self,embed_dim,num_heads,ff_dim,dropout_rate):
    super().__init__()
    self.attention=MultiHeadSelfAttention(embed_dim,num_heads,dropout_rate)
    self.feed_forward=PositionwiseFeedForward(embed_dim,ff_dim,dropout_rate)
    self.norm1=nn.LayerNorm(embed_dim)
    self.norm2=nn.LayerNorm(embed_dim)
    self.dropout1=nn.Dropout(dropout_rate)
    self.dropout2=nn.Dropout(dropout_rate)
  def forward(self,x,mask=None):
    attn_output=self.attention(x,x,x,mask)
    x=self.norm1(x+self.dropout1(attn_output))
    ff_output=self.feed_forward(x)
    x=self.norm2(x+self.dropout2(ff_output))
    return x

class NextSentencePredictionHead(nn.Module):
  def __init__(self,embed_dim):
    super().__init__()
    self.classifier=nn.Linear(embed_dim,2)
  def forward(self,cls_output):
    return self.classifier(cls_output)
  
class MaskedLanguageModelHead(nn.Module):
  def __init__(self,embed_dim,vocab_size):
    super().__init__()
    self.dense=nn.Linear(embed_dim,embed_dim)
    self.gelu=nn.GELU()
    self.norm=nn.LayerNorm(embed_dim)
    self.decoder=nn.Linear(embed_dim,vocab_size)
  def forward(self,x):
    return self.decoder(self.norm(self.gelu(self.dense(x))))