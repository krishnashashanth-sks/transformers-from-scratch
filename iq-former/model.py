from layers import *
import copy

class iQTransformer(nn.Module):
  def __init__(self,src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, num_heads=8, dropout=0.1, num_bits=8):
    super(iQTransformer,self).__init__()
    c = copy.deepcopy
    attn = QuantizedMultiHeadAttention(d_model, num_heads, dropout, num_bits)
    ff = QuantizedFeedForward(d_model, d_ff, dropout, num_bits)
    position = PositionalEncoding(d_model, max_len=5000)
    self.encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    self.decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    self.src_embed=nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    self.tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    self.generator=Generator(d_model, tgt_vocab)
  def forward(self,src,tgt,src_mask,tgt_mask):
    return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
  def encode(self,src,src_mask):
    return self.encoder(self.src_embed(src),src_mask)
  def decode(self,memory,src_mask,tgt,tgt_mask):
    return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)