import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,Conv1D,MaxPooling1D,Bidirectional,LSTM,
    GlobalAveragePooling1D,Dense
)
class GenieArchitecture(tf.keras.Model):
  def __init__(self,vocab_size,embedding_dim,max_sequence_length,conv_filters,conv_kernel_size,lstm_units):
    super().__init__()
    self.embedding=Embedding(
        vocab_size,
        embedding_dim
    )
    self.conv1d=Conv1D(
        conv_filters,conv_kernel_size,activation='relu'
    )
    self.max_pooling1d=MaxPooling1D(pool_size=2)
    self.bidirectional_lstm=Bidirectional(LSTM(
        lstm_units,
        return_sequences=True
    ))
    self.embedding_dim=embedding_dim
    self.lstm_units=lstm_units

    self.global_average_pooling1d=GlobalAveragePooling1D()
    self.dense_output=Dense(1,activation='sigmoid')
  def build(self,input_shape):
    conv_input_shape=(input_shape[0],input_shape[1],self.embedding_dim)
    self.conv1d.build(conv_input_shape)
    conv_output_length=input_shape[1]-self.conv1d.kernel_size[0]+1
    pool_input_shape=(input_shape[0],conv_output_length,self.conv1d.filters)
    self.max_pooling1d.build(pool_input_shape)
    pool_output_length=conv_output_length//self.max_pooling1d.pool_size[0]
    lstm_input_shape=(input_shape[0],pool_output_length,self.conv1d.filters)
    self.bidirectional_lstm.build(lstm_input_shape)
    global_pool_input_shape=(input_shape[0],pool_output_length,2*self.lstm_units)
    self.global_average_pooling1d.build(global_pool_input_shape)
    dense_input_shape=(input_shape[0],2*self.lstm_units)
    self.dense_output.build(dense_input_shape)
    self.built=True
  def call(self,inputs):
    x = self.embedding(inputs)
    x = self.conv1d(x)
    x = self.max_pooling1d(x)
    x = self.bidirectional_lstm(x)
    x = self.global_average_pooling1d(x)
    return self.dense_output(x)