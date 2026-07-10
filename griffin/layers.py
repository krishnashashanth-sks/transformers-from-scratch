import tensorflow as tf
from tensorflow.keras import layers

# --- CustomGRU Layer --- #
class CustomGRU(layers.Layer):
  def __init__(self, units, **kwargs):
    super(CustomGRU, self).__init__(**kwargs)
    self.units = units
    self.dense_iz = layers.Dense(units, activation=None, use_bias=False, name='input_to_update_gate')
    self.dense_hz = layers.Dense(units, activation=None, use_bias=True, name='hidden_to_update_gate')
    self.dense_ir = layers.Dense(units, activation=None, use_bias=False, name='input_to_reset_gate')
    self.dense_hr = layers.Dense(units, activation=None, use_bias=True, name='hidden_to_reset_gate')
    self.dense_ih = layers.Dense(units, activation=None, use_bias=False, name='input_to_candiate_gate')
    self.dense_hh = layers.Dense(units, activation=None, use_bias=True, name='hidden_to_candiate_gate')

  def call(self, inputs, states):
    h_prev = states[0]
    z_t_input = self.dense_iz(inputs)
    z_t_hidden = self.dense_hz(h_prev)
    z_t = tf.sigmoid(z_t_input + z_t_hidden)

    r_t_input = self.dense_ir(inputs)
    r_t_hidden = self.dense_hr(h_prev)
    r_t = tf.sigmoid(r_t_input + r_t_hidden) # Corrected: added + operator

    h_candiate_input = self.dense_ih(inputs)
    h_candiate_hidden = self.dense_hh(r_t * h_prev)
    h_t_candiate = tf.tanh(h_candiate_input + h_candiate_hidden)

    h_t = (1 - z_t) * h_t_candiate + z_t * h_prev
    return h_t, [h_t]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      batch_size = tf.shape(inputs)[0]
      dtype = inputs.dtype
    return [tf.zeros((batch_size, self.units), dtype=dtype)]

# --- MultiHeadSelfAttention Layer --- #
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

        self.depth = key_dim

        self.wq = tf.keras.layers.Dense(key_dim * num_heads, use_bias=False, name='query_projection')
        self.wk = tf.keras.layers.Dense(key_dim * num_heads, use_bias=False, name='key_projection')
        self.wv = tf.keras.layers.Dense(key_dim * num_heads, use_bias=False, name='value_projection')

        self.dense = tf.keras.layers.Dense(key_dim * num_heads, name='output_projection')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, training=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, num_heads * depth)
        k = self.wk(k)  # (batch_size, seq_len, num_heads * depth)
        v = self.wv(v)  # (batch_size, seq_len, num_heads * depth)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.num_heads * self.depth))  # (batch_size, seq_len_q, num_heads * depth)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, model_dim)
        output = self.dropout(output, training=training)

        return output, attention_weights

# --- FeedForwardNetwork Layer --- #
class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.0, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.d_model = d_model # Output dimension
        self.dff = dff # Inner dimension of the feed forward network

        self.dense1 = tf.keras.layers.Dense(dff, activation='relu', name='ffn_dense1')
        self.dense2 = tf.keras.layers.Dense(d_model, name='ffn_dense2')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # (batch_size, seq_len, dff)
        x = self.dense1(inputs)
        # (batch_size, seq_len, d_model)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

# --- GriffinBlock Layer --- #
class GriffinBlock(layers.Layer):
  def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
    super(GriffinBlock, self).__init__(**kwargs)
    self.d_model = d_model
    # Corrected: MultiHEadSelfAttention to MultiHeadSelfAttention
    # Corrected: num_nodes to num_heads, drop_rate to dropout_rate
    self.mha = MultiHeadSelfAttention(num_heads, d_model // num_heads, dropout_rate)
    self.gru = CustomGRU(d_model)
    self.ffn = FeedForwardNetwork(d_model, dff, dropout_rate)

    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = layers.Dropout(dropout_rate)
    self.dropout2 = layers.Dropout(dropout_rate)
    self.dropout3 = layers.Dropout(dropout_rate)

  def call(self, inputs, training=False, state_gru=None):
    # Multi-head attention
    attn_output, _ = self.mha(inputs, inputs, inputs, training=training)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output)

    # GRU processing
    gru_outputs = []
    if state_gru is None:
      initial_state_gru = self.gru.get_initial_state(inputs=out1[:, 0, :])
      current_gru_state = initial_state_gru[0]
    else:
      current_gru_state = state_gru

    for t in tf.range(tf.shape(out1)[1]): # Use tf.range for graph mode compatibility
      current_input_t = out1[:, t, :]
      # Corrected: self.gri to self.gru, new_stated_t to new_state_t
      output_t, new_state_t = self.gru(current_input_t, [current_gru_state])
      gru_outputs.append(output_t)
      current_gru_state = new_state_t[0]

    gru_output = tf.stack(gru_outputs, axis=1)
    gru_output = self.dropout2(gru_output, training=training)
    # Corrected: out1 to out2 for clarity and correct residual connection
    out2 = self.layernorm2(out1 + gru_output)

    # Feed forward network
    ffn_output = self.ffn(out2, training=training)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(out2 + ffn_output)

    return out3, current_gru_state

