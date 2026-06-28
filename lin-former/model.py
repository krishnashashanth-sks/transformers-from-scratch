import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers import LinformerEncoderBlock
class Linformer(keras.Model):
  def __init__(self,vocab_size,sequence_length,d_model,k_dim,v_dim,num_heads,ff_dim,num_blocks,dropout_rate=0.1,num_classes=None,**kwargs):
    super().__init__(**kwargs)
    self.d_model=d_model
    self.sequence_length=sequence_length
    self.token_embedding=layers.Embedding(vocab_size,d_model)
    self.position_embedding=layers.Embedding(sequence_length,d_model)
    self.encoder_blocks=[
        LinformerEncoderBlock(d_model,k_dim,v_dim,num_heads,ff_dim,dropout_rate)
        for _ in range(num_blocks)
    ]
    self.dropout_final=layers.Dropout(dropout_rate)
    self.final_layer=None
    if num_classes:
      self.final_layer=layers.Dense(num_classes,activation='softmax')

  def build(self, input_shape):
    # Input shape for the Linformer model is (batch_size, sequence_length)
    # Embeddings will take (batch_size, sequence_length)
    self.token_embedding.build(input_shape)
    self.position_embedding.build(input_shape)

    # After embeddings, the shape becomes (batch_size, sequence_length, d_model)
    encoder_input_shape = tf.TensorShape([input_shape[0], input_shape[1], self.d_model])
    for encoder_block in self.encoder_blocks:
      encoder_block.build(encoder_input_shape)

    if self.final_layer:
      # After tf.reduce_mean, the shape for the final_layer becomes (batch_size, d_model)
      final_layer_input_shape = tf.TensorShape([input_shape[0], self.d_model])
      self.final_layer.build(final_layer_input_shape)

    super().build(input_shape) # Call the base Model's build method last

  def call(self,inputs,training=False):
    positions=tf.range(start=0,limit=self.sequence_length,delta=1)
    positions=self.position_embedding(positions)
    x=self.token_embedding(inputs)
    x=x+positions
    x=self.dropout_final(x,training=training)
    for encoder_block in self.encoder_blocks:
      x=encoder_block(x,training=training)
    if self.final_layer:
      x=tf.reduce_mean(x,axis=1)
      x=self.final_layer(x)
    return x