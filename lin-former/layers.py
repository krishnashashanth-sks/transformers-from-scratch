from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

class LinearAttention(layers.Layer):
  def __init__(self,d_model,k_dim,v_dim,h=8,dropout_rate=0.1,**kwargs):
    super().__init__(**kwargs)
    self.d_model=d_model
    self.k_dim=k_dim
    self.v_dim=v_dim
    self.h=h
    self.dropout_rate=dropout_rate
    self.query_projection=layers.Dense(d_model)
    self.key_projection=layers.Dense(d_model)
    self.value_projection=layers.Dense(d_model)
    # Initialize key and value sequence reducers here
    self.key_seq_reducer=layers.Dense(self.k_dim,use_bias=False)
    self.value_seq_reducer=layers.Dense(self.v_dim,use_bias=False)
    self.E=self.add_weight(
        name='projection_E',
        shape=(self.k_dim,self.d_model),
        initializer='glorot_uniform',
        trainable=True
    )
    self.F=self.add_weight(
        name='projection_F',
        shape=(self.v_dim,self.d_model),
        initializer='glorot_uniform',
        trainable=True
    )
    self.output_projection=layers.Dense(d_model)
    self.dropout=layers.Dropout(dropout_rate)

  def build(self, input_shape):
    # input_shape here is (batch_size, sequence_length, d_model)
    # The dense layers will receive an input of shape (batch_size, sequence_length, d_model)
    self.query_projection.build(input_shape)
    self.key_projection.build(input_shape)
    self.value_projection.build(input_shape)

    # Build key_seq_reducer and value_seq_reducer
    # The input shape to these layers will be (batch_size, num_heads, sequence_length, d_model//h)
    # We need to consider the 'd_model//h' part as the last dimension
    k_v_input_shape = tf.TensorShape([input_shape[0], self.h, input_shape[1], self.d_model // self.h])
    self.key_seq_reducer.build(k_v_input_shape)
    self.value_seq_reducer.build(k_v_input_shape)

    self.output_projection.build(input_shape)
    # The dropout layer's build method is usually called implicitly
    # The weight 'E' and 'F' are already built by add_weight
    super().build(input_shape)

  def call(self,inputs,training=False):
    batch_size=tf.shape(inputs)[0]
    sequence_length=tf.shape(inputs)[1]
    q=self.query_projection(inputs)
    k=self.key_projection(inputs)
    v=self.value_projection(inputs)
    q=tf.reshape(q,(batch_size,sequence_length,self.h,self.d_model//self.h))
    k=tf.reshape(k,(batch_size,sequence_length,self.h,self.d_model//self.h))
    v=tf.reshape(v,(batch_size,sequence_length,self.h,self.d_model//self.h))
    q=tf.transpose(q,perm=[0,2,1,3])
    k=tf.transpose(k,perm=[0,2,1,3])
    v=tf.transpose(v,perm=[0,2,1,3])
    # Use the pre-instantiated layers
    k_proj=self.key_seq_reducer(k)
    v_proj=self.value_seq_reducer(v)
    attention_scores=tf.matmul(q,k_proj,transpose_b=True)
    attention_scores=attention_scores/tf.math.sqrt(tf.cast(self.d_model//self.h,tf.float32))
    attention_weights=tf.nn.softmax(attention_scores,axis=-1)
    attention_weights=self.dropout(attention_weights,training=training)
    attention_output=tf.matmul(attention_weights,v_proj)
    attention_output=tf.transpose(attention_output,perm=[0,2,1,3])
    concat_attention=tf.reshape(attention_output,(batch_size,sequence_length,self.d_model))
    output=self.output_projection(concat_attention)
    return output

class LinformerEncoderBlock(layers.Layer):
  def __init__(self,d_model,k_dim,v_dim,h,ff_dim,dropout_rate=0.1,**kwargs):
    super().__init__(**kwargs)
    self.d_model=d_model
    self.k_dim=k_dim
    self.v_dim=v_dim
    self.h=h
    self.ff_dim=ff_dim
    self.linear_attention=LinearAttention(d_model,k_dim,v_dim,h,dropout_rate)
    self.layernorm1=layers.LayerNormalization(epsilon=1e-6)
    self.dropout1=layers.Dropout(dropout_rate)
    self.ffn=keras.Sequential(
        [
            layers.Dense(ff_dim,activation='relu'),
            layers.Dense(d_model)
        ]
    )
    self.layernorm2=layers.LayerNormalization(epsilon=1e-6)
    self.dropout2=layers.Dropout(dropout_rate)

  def build(self, input_shape):
    self.linear_attention.build(input_shape)
    self.layernorm1.build(input_shape)
    self.dropout1.build(input_shape)
    self.ffn.build(input_shape)
    self.layernorm2.build(input_shape)
    self.dropout2.build(input_shape)
    super().build(input_shape)

  def call(self,inputs,training=False):
    attn_output=self.linear_attention(inputs,training=training)
    attn_output=self.dropout1(attn_output,training=training)
    out1=self.layernorm1(inputs+attn_output)
    ffn_output=self.ffn(out1)
    ffn_output=self.dropout2(ffn_output,training=training)
    return self.layernorm2(out1+ffn_output)