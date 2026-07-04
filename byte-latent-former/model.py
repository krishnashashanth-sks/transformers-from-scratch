import torch.nn as nn
from layers import EncoderTransformer,LatentSpaceModule,DecoderTransformer,OutputHead

class ByteLatentTransformer(nn.Module):
  def __init__(self,vocab_size,d_model,num_heads,d_ff,num_encoder_layers,num_decoder_layers,d_latent,dropout_rate=0.1,max_len=5000,eps=1e-6):
    super(ByteLatentTransformer,self).__init__()
    self.encoder=EncoderTransformer(
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_encoder_layers,
        dropout_rate,
        max_len,
        eps
    )
    self.latent_module=LatentSpaceModule(d_model,d_latent)
    self.decoder=DecoderTransformer(
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_decoder_layers,
        d_latent,
        dropout_rate,
        max_len,eps
    )
    self.output_head=OutputHead(d_model,vocab_size)

  def forward(self,src,tgt,src_mask=None,tgt_mask=None):
    encoder_output=self.encoder(src,src_mask)
    z,mu,log_var=self.latent_module(encoder_output)
    decoder_output=self.decoder(tgt,z,tgt_mask)
    output_logits=self.output_head(decoder_output)
    return output_logits,mu,log_var
