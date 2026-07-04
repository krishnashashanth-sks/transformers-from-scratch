from tensorflow import keras
from tensorflow.keras import layers
from layers import TokenAndPositionEmbedding,TransformerBlock

class GatoLikeTransformer(keras.Model):
  def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_transformer_blocks, rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.token_pos_embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    self.transformer_blocks = [
        TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        for _ in range(num_transformer_blocks)
    ]
    self.final_layer_norm = layers.LayerNormalization(epsilon=1e-6)
    # Add a final dense layer to project to vocabulary size for token prediction
    self.output_dense = layers.Dense(vocab_size)

  def call(self, inputs, training=False):
    # Inputs should be a sequence of token IDs (e.g., from different modalities)
    x = self.token_pos_embedding(inputs)
    for transformer_block in self.transformer_blocks:
      x = transformer_block(x, training=training)
    x = self.final_layer_norm(x)
    # Project the normalized output to the vocabulary size
    return self.output_dense(x)