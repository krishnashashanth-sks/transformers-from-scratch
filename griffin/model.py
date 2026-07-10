import tensorflow as tf
from tensorflow.keras import layers
from layers import GriffinBlock

# --- Full Griffin Model (Example) --- #
class GriffinModel(tf.keras.Model):
    def __init__(self, num_blocks, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1, **kwargs):
        super(GriffinModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self._get_positional_encoding(10000, d_model)
        self.dropout = layers.Dropout(dropout_rate)

        self.griffin_blocks = [
            GriffinBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_blocks)
        ]

        self.final_layer = layers.Dense(target_vocab_size)

    def _get_positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                      tf.range(d_model, dtype=tf.float32)[tf.newaxis, :])

        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i + 1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return pos_encoding

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        attention_states = [None] * self.num_blocks # Placeholder for GRU states across blocks

        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i, griffin_block in enumerate(self.griffin_blocks):
            # The first block will initialize its GRU state, subsequent blocks can carry forward if needed
            # For simplicity here, we re-initialize per block per sequence, but it can be chained.
            x, attention_states[i] = griffin_block(x, training=training)

        output = self.final_layer(x)
        return output
