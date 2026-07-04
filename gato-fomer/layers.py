import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
  def __init__(self,embed_dim,num_heads=8,**kwargs):
    super().__init__(**kwargs)
    self.embed_dim=embed_dim
    self.num_heads=num_heads
    if embed_dim % num_heads !=0:
      raise ValueError(
          f"embed_dim ={embed_dim} should be divisible by num_heads ={num_heads}"
      )
    self.proj_dim=embed_dim//num_heads
    self.query_dense=layers.Dense(embed_dim)
    self.key_dense=layers.Dense(embed_dim)
    self.value_dense=layers.Dense(embed_dim)
    self.combine_heads=layers.Dense(embed_dim)
  def attention(self,query,key,value):
    score=tf.matmul(query,key,transpose_b=True)
    dim_key=tf.cast(tf.shape(key)[-1],tf.float32)
    scaled_score=score/tf.math.sqrt(dim_key)
    attention_scores=tf.nn.softmax(scaled_score,axis=-1)
    attention_output_values=tf.matmul(attention_scores,value)
    return attention_scores,attention_output_values
  def separate_heads(self,x,batch_size):
    x=tf.reshape(x,(batch_size,-1,self.num_heads,self.proj_dim))
    return tf.transpose(x,perm=[0,2,1,3])
  def call(self,inputs):
    batch_size=tf.shape(inputs)[0]
    sequence_length=tf.shape(inputs)[1] # Explicitly capture sequence length
    query=self.query_dense(inputs)
    key=self.key_dense(inputs)
    value=self.value_dense(inputs)
    query=self.separate_heads(query,batch_size)
    value=self.separate_heads(value,batch_size)
    key=self.separate_heads(key,batch_size)
    # Explicitly unpack the returned values
    attention_scores,attention_output_values=self.attention(query,key,value)
    attention_output_values=tf.transpose(attention_output_values,perm=[0,2,1,3])
    # Use the captured sequence_length to ensure correct reshaping
    concat_attention=tf.reshape(attention_output_values,(batch_size,sequence_length,self.embed_dim))
    return self.combine_heads(concat_attention)
  
class TransformerBlock(layers.Layer):
  def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1,**kwargs):
    super().__init__(**kwargs)
    self.attn=MultiHeadAttention(embed_dim,num_heads)
    self.ffn=keras.Sequential(
        [
            layers.Dense(ff_dim,activation='relu'),
            layers.Dense(embed_dim)
        ]
    )
    self.layernorm1=layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2=layers.LayerNormalization(epsilon=1e-6)
    self.dropout1=layers.Dropout(rate)
    self.dropout2=layers.Dropout(rate)
  def call(self,inputs,training):
    attn_output=self.attn(inputs) # Corrected from self.att
    attn_output=self.dropout1(attn_output,training=training)
    out1=self.layernorm1(inputs+attn_output)
    ffn_output=self.ffn(out1)
    ffn_output=self.dropout2(ffn_output,training=training)
    return self.layernorm2(out1+ffn_output)
  
class TokenAndPositionEmbedding(layers.Layer):
  def __init__(self,maxlen,vocab_size,embed_dim,**kwargs):
    super().__init__(**kwargs)
    self.token_emb=layers.Embedding(input_dim=vocab_size,output_dim=embed_dim)
    self.pos_emb=layers.Embedding(input_dim=maxlen,output_dim=embed_dim)
    self.maxlen=maxlen
  def call(self,x):
    positions=tf.range(start=0,limit=self.maxlen,delta=1)
    positions=self.pos_emb(positions)
    x=self.token_emb(x)
    return x+positions