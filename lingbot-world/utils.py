import tensorflow as tf
import numpy as np
from main import PAD_ID

def create_padding_mask(seq):
    # seq is of shape (batch_size, seq_len)
    # Where padding_id is typically 0, and was defined as PAD_ID in previous cells
    seq = tf.cast(tf.math.equal(seq, PAD_ID), tf.float32)

    # Add extra dimensions to add the padding
    # to the attention logits. For example, a sequence of length 5 will be masked as:
    # [[0., 0., 0., 0., 1.]] if the last token is padding.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i + 1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def pad_sequences(sequences, max_len, padding_value):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [padding_value] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len] # Truncate if longer than max_len
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

def tokenize_and_create_sequences(data, char_to_id, SOS_ID, EOS_ID, UNK_ID):
    input_sequences = []
    target_sequences = []

    # Helper function to convert a string to a list of IDs
    def text_to_ids(text):
        return [char_to_id.get(char, UNK_ID) for char in text]

    # Split data into input (command/observation) and target (response/next state)
    # For this synthetic dataset, we'll assume a simple input-output pairing.
    # For a real LingBot World, this might involve more complex state transitions.
    for i in range(len(data) - 1):
        # Input is the current sentence
        encoder_input_text = data[i]
        encoder_input_ids = text_to_ids(encoder_input_text)
        input_sequences.append(encoder_input_ids)

        # Target is the next sentence, prepended with SOS and appended with EOS
        decoder_target_text = data[i+1]
        decoder_target_ids = [SOS_ID] + text_to_ids(decoder_target_text) + [EOS_ID]
        target_sequences.append(decoder_target_ids)

    # For the last item, we'll use it as an input but won't have a subsequent target
    # If we need a target for the last input, we'd need to define it or pad it differently.
    # For simplicity, we'll just process N-1 pairs for now.

    return input_sequences, target_sequences

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
