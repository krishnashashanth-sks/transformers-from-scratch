import tensorflow as tf
from layers import Encoder,Decoder
from utils import create_look_ahead_mask

# ==============================================================================
#  Informer Model
# ==============================================================================
class Informer(tf.keras.Model):
    def __init__(self,
                 enc_seq_len,
                 label_len,
                 pred_len,
                 output_attention=False,
                 enc_in=7,
                 dec_in=7,
                 c_out=7,
                 d_model=512,
                 num_heads=8,
                 e_layers=3,
                 d_layers=2,
                 d_ff=2048,
                 dropout_rate=0.1,
                 attn_factor=5,
                 distil=True,
                 activation='gelu',
                 **kwargs):

        super(Informer, self).__init__(**kwargs)
        self.enc_seq_len = enc_seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.d_model = d_model

        self.encoder = Encoder(
            num_layers=e_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            factor=attn_factor,
            distill_layers=[distil] * e_layers # Apply distilling to all encoder layers if distil is True
        )

        self.decoder = Decoder(
            num_layers=d_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate
        )

        self.projection = tf.keras.layers.Dense(c_out, activation='linear')

    def call(self, inputs, training=False):
        enc_input, dec_input, enc_time_features, dec_time_features = inputs

        batch_size = tf.shape(enc_input)[0]

        encoder_padding_mask = None
        enc_output = self.encoder(enc_input, enc_time_features, attn_mask=encoder_padding_mask, training=training)

        decoder_seq_len = tf.shape(dec_input)[1]
        look_ahead_mask = create_look_ahead_mask(decoder_seq_len)

        decoder_padding_mask = None
        cross_attention_padding_mask = encoder_padding_mask

        dec_output = self.decoder(
            dec_input,
            enc_output,
            dec_time_features,
            look_ahead_mask,
            cross_attention_padding_mask,
            training=training
        )

        output = self.projection(dec_output[:, -self.pred_len:, :])

        return output