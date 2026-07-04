import tensorflow as tf
from utils import create_look_ahead_mask,create_padding_mask
from main import UNK_ID

def predict_sentence(input_sentence, transformer, char_to_id, id_to_char, SOS_ID, EOS_ID, PAD_ID, max_input_len, max_target_len):
    # 1. Prepare the input sentence
    # Convert input string to IDs
    input_ids = [char_to_id.get(char, UNK_ID) for char in input_sentence]

    # Pad the input sequence to max_input_len
    input_padded = input_ids + [PAD_ID] * (max_input_len - len(input_ids))
    input_tensor = tf.expand_dims(tf.constant(input_padded, dtype=tf.int64), axis=0) # Add batch dimension

    # 2. Initialize the decoder input with the start-of-sequence token
    decoder_input = tf.expand_dims(tf.constant([SOS_ID], dtype=tf.int64), axis=0)

    # 3. Generate output sequence token by token
    output_sentence = []
    for i in range(max_target_len):
        # Create padding masks for encoder and decoder
        enc_padding_mask = create_padding_mask(input_tensor)
        look_ahead_mask = create_look_ahead_mask(tf.shape(decoder_input)[1])
        dec_padding_mask = create_padding_mask(input_tensor) # Cross-attention mask

        # Get predictions from the transformer
        predictions, _ = transformer(
            input_tensor,
            decoder_input,
            training=False, # Always False for inference
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=look_ahead_mask,
            dec_padding_mask=dec_padding_mask
        )

        # Select the last token's logits and get the predicted ID
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int64)

        # Append the predicted ID to the decoder input for the next step
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

        # If the predicted ID is the end-of-sequence token, stop generation
        if predicted_id == EOS_ID:
            break

        output_sentence.append(id_to_char.get(predicted_id.numpy()[0][0], '')) # Convert ID to character

    # Join characters to form the output sentence
    return ''.join(output_sentence)

print("Prediction function defined.")