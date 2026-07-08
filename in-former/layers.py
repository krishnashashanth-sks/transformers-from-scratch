import tensorflow as tf
import numpy as np

# ==============================================================================
# 1. ProbSparse Self-Attention Class
# ==============================================================================
class ProbSparseSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, factor=5, **kwargs):
        super(ProbSparseSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of key, query, value per head
        self.factor = factor # For L_q calculation

        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.output_dense = tf.keras.layers.Dense(d_model)

    def build(self, input_shape):
        super(ProbSparseSelfAttention, self).build(input_shape)

    def _split_heads(self, inputs, batch_size):
        # Reshape (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, d_k)
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.d_k))
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def _calculate_M_statistic(self, queries, keys):
        # queries shape: (batch_size, num_heads, seq_len_Q, d_k)
        # keys shape: (batch_size, num_heads, seq_len_K, d_k)

        # Calculate dot product of queries and keys (Q.K^T)
        qk_scores = tf.einsum('bnqd,bnkd->bnqk', queries, keys)

        # Sum over the keys dimension (dim=3)
        M_statistic = tf.reduce_sum(qk_scores, axis=-1)

        # Normalize by log(seq_len_K)
        seq_len_K = tf.shape(keys)[2]
        M_statistic /= tf.math.log(tf.cast(seq_len_K, dtype=tf.float32))

        return M_statistic

    def call(self, inputs, attn_mask=None, training=False):
        query, key, value = inputs # inputs is a tuple/list of (query, key, value)

        batch_size = tf.shape(query)[0]

        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split into multiple heads
        queries = self._split_heads(query, batch_size)
        keys = self._split_heads(key, batch_size)
        values = self._split_heads(value, batch_size)

        seq_len_Q = tf.shape(queries)[2]
        seq_len_K = tf.shape(keys)[2]

        # Calculate M-statistic
        M_statistic = self._calculate_M_statistic(queries, keys)

        # Determine L_q (number of selected queries)
        L_q = tf.minimum(self.factor * tf.cast(tf.math.log(tf.cast(seq_len_Q, dtype=tf.float32)), dtype=tf.int32), seq_len_Q)

        # Select top L_q queries based on M-statistic
        _, top_Lq_indices = tf.math.top_k(M_statistic, k=L_q, sorted=True)

        # Expand indices for gathering queries
        batch_indices = tf.range(batch_size)[:, None, None]
        head_indices = tf.range(self.num_heads)[None, :, None]

        # For tf.gather_nd, indices must be (batch_size * num_heads * L_q, 3)
        gather_indices = tf.stack([
            tf.tile(batch_indices, [1, self.num_heads, L_q]),
            tf.tile(head_indices, [batch_size, 1, L_q]),
            top_Lq_indices
        ], axis=-1)
        gather_indices = tf.reshape(gather_indices, [-1, 3])

        # Gather the top L_q queries from the original queries tensor
        queries_top = tf.gather_nd(queries, gather_indices)
        queries_top = tf.reshape(queries_top, (batch_size, self.num_heads, L_q, self.d_k))

        # 1. Calculate sparse attention scores
        attention_scores = tf.einsum('bnqd,bnkd->bnqk', queries_top, keys)
        attention_scores /= tf.math.sqrt(tf.cast(self.d_k, dtype=tf.float32))

        # 2. Apply attention mask if provided, then softmax
        if attn_mask is not None:
            attention_scores = tf.where(tf.equal(attn_mask, 0), -1e9, attention_scores)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # 3. Multiply with values to get attended values for selected queries
        queries_top_attended_values = tf.einsum('bnqk,bnkd->bnqd', attention_weights, values)

        # 4. Reconstruct the full attention output for all queries
        full_attended_values = tf.zeros((batch_size, self.num_heads, seq_len_Q, self.d_k), dtype=queries.dtype)

        scatter_indices = gather_indices
        updates = tf.reshape(queries_top_attended_values, (-1, self.d_k))

        full_attended_values = tf.tensor_scatter_nd_update(
            tensor=tf.cast(full_attended_values, dtype=updates.dtype),
            indices=scatter_indices,
            updates=updates
        )

        # 5. Reshape and concatenate heads
        full_attended_values = tf.transpose(full_attended_values, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(full_attended_values, (batch_size, seq_len_Q, self.d_model))

        # 6. Final linear projection
        output = self.output_dense(concat_attention)

        return output


# ==============================================================================
# 2. Self-Attention Distilling Class
# ==============================================================================
class SelfAttentionDistilling(tf.keras.layers.Layer):
  def __init__(self, c_out, dropout_rate=0.1, **kwargs):
    super(SelfAttentionDistilling, self).__init__(**kwargs)
    self.conv1d = tf.keras.layers.Conv1D(
        filters=c_out,
        kernel_size=3,
        padding='same',
        strides=1,
        activation='relu'
    )
    self.max_pool1d = tf.keras.layers.MaxPool1D(
        pool_size=3,
        strides=2,
        padding='same'
    )
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training=False):
    x = self.conv1d(inputs)
    x = self.dropout(x, training=training)
    return self.max_pool1d(x)


# ==============================================================================
# 3. Encoder Layer
# ==============================================================================
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, factor=5, distill=True, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.distill = distill

        # ProbSparse Self-Attention
        self.self_attention = ProbSparseSelfAttention(d_model, num_heads, factor=factor)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed-Forward Network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Self-Attention Distilling (optional)
        if self.distill:
            self.distilling_layer = SelfAttentionDistilling(c_out=d_model, dropout_rate=dropout_rate)

    def call(self, inputs, attn_mask=None, training=False):
        # Self-Attention Block
        attn_output = self.self_attention(inputs=[inputs, inputs, inputs], attn_mask=attn_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        # Feed-Forward Network Block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)

        # Optional Distilling
        if self.distill:
            output = self.distilling_layer(out2, training=training)
        else:
            output = out2

        return output


# ==============================================================================
# 4. Decoder Layer
# ==============================================================================
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        # Masked Multi-Head Self-Attention
        self.masked_self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Cross-Attention
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed-Forward Network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_output, look_ahead_mask, padding_mask, training=False):
        # Masked Multi-Head Self-Attention Block
        attn1_output = self.masked_self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=look_ahead_mask,
            training=training
        )
        attn1_output = self.dropout1(attn1_output, training=training)
        out1 = self.norm1(inputs + attn1_output)

        # Cross-Attention Block
        attn2_output = self.cross_attention(
            query=out1,
            value=enc_output,
            key=enc_output,
            attention_mask=padding_mask,
            training=training
        )
        attn2_output = self.dropout2(attn2_output, training=training)
        out2 = self.norm2(out1 + attn2_output)

        # Feed-Forward Network Block
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.norm3(out2 + ffn_output)

        return out3




# ==============================================================================
# 5. Embedding Layers
# ==============================================================================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_seq_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pos_encoding = tf.cast(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


class TimeFeatureEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model, **kwargs):
    super(TimeFeatureEmbedding, self).__init__(**kwargs)
    self.d_model = d_model
    self.minute_embed = tf.keras.layers.Embedding(60, d_model)
    self.hour_embed = tf.keras.layers.Embedding(24, d_model)
    self.day_embed = tf.keras.layers.Embedding(32, d_model)
    self.weekday_embed = tf.keras.layers.Embedding(7, d_model)
    self.month_embed = tf.keras.layers.Embedding(13, d_model)
    self.year_embed = tf.keras.layers.Embedding(20, d_model)

  def call(self, inputs):
    minute_x = self.minute_embed(inputs[:, :, 0])
    hour_x = self.hour_embed(inputs[:, :, 1])
    day_x = self.day_embed(inputs[:, :, 2])
    weekday_x = self.weekday_embed(inputs[:, :, 3])
    month_x = self.month_embed(inputs[:, :, 4])
    year_x = self.year_embed(inputs[:, :, 5])
    return minute_x + hour_x + day_x + weekday_x + month_x + year_x


# ==============================================================================
# 6. Encoder and Decoder Stacks
# ==============================================================================
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_rate, factor, distill_layers=None, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers

        self.value_embedding = tf.keras.layers.Dense(d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.time_feature_embedding = TimeFeatureEmbedding(d_model=d_model)

        self.encoder_layers = []
        for i in range(num_layers):
            distill = distill_layers[i] if distill_layers is not None and i < len(distill_layers) else False
            self.encoder_layers.append(EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                factor=factor,
                distill=distill
            ))

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, enc_input, enc_time_features, attn_mask=None, training=False):
        x = self.value_embedding(enc_input)
        x = self.positional_encoding(x)
        x += self.time_feature_embedding(enc_time_features)

        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x, attn_mask=attn_mask, training=training)

        output = self.norm(x)

        return output


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers

        self.value_embedding = tf.keras.layers.Dense(d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.time_feature_embedding = TimeFeatureEmbedding(d_model=d_model)

        self.decoder_layers = []
        for _ in range(num_layers):
            self.decoder_layers.append(DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate
            ))

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, dec_input, enc_output, dec_time_features, look_ahead_mask, padding_mask, training=False):
        x = self.value_embedding(dec_input)
        x = self.positional_encoding(x)
        x += self.time_feature_embedding(dec_time_features)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, look_ahead_mask, padding_mask, training=training)

        output = self.norm(x)

        return output