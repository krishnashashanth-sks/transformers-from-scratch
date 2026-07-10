import tensorflow as tf

# --- Inference Function --- #
def inference(model, input_sequence, max_length=50):
    # input_sequence: a tensor of shape (1, seq_len) with starting tokens
    output_sequence = tf.identity(input_sequence)

    for _ in range(max_length - tf.shape(input_sequence)[1]):
        predictions = model(output_sequence, training=False)
        # Get the last token's predictions and sample the next token
        last_token_logits = predictions[:, -1, :]
        next_token = tf.argmax(last_token_logits, axis=-1)[:, tf.newaxis]
        output_sequence = tf.concat([output_sequence, next_token], axis=-1)

    return output_sequence