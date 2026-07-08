import tensorflow as tf

# Function to generate a single Informer data sample
def create_informer_data_sample(data_series, time_features_series, start_idx, seq_len, label_len, pred_len):
    encoder_input = data_series[start_idx : start_idx + seq_len]
    encoder_time_features = time_features_series[start_idx : start_idx + seq_len]

    decoder_input_start_idx = start_idx + seq_len - label_len
    decoder_input_end_idx = start_idx + seq_len + pred_len
    decoder_input = data_series[decoder_input_start_idx : decoder_input_end_idx]
    decoder_time_features = time_features_series[decoder_input_start_idx : decoder_input_end_idx]

    target_output = data_series[start_idx + seq_len : start_idx + seq_len + pred_len]

    return encoder_input, decoder_input, encoder_time_features, decoder_time_features, target_output

# ==============================================================================
#  Mask Generation Functions
# ==============================================================================
def create_padding_mask(seq):
    """
    Generates a boolean mask for padded sequences.

    Args:
        seq: A tensor of shape (batch_size, seq_len).
             Padding values are typically 0.

    Returns:
        A boolean tensor of shape (batch_size, 1, 1, seq_len)
        with 'True' at padding positions.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_len):
    """
    Generates a causal (look-ahead) mask for decoder self-attention.

    Args:
        seq_len: An integer, the length of the sequence.

    Returns:
        A boolean tensor of shape (1, 1, seq_len, seq_len)
        with 'True' at positions to be masked (future positions).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]
