import tensorflow as tf
from main import PAD_ID

def accuracy_function(real, pred):
    # Ensure pred is not None and has a shape
    if pred is None or tf.shape(pred)[0] == 0:
        return tf.constant(0.0)

    # Get the predicted ID for each token
    # tf.argmax returns the index of the largest value across axes
    predictions = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)

    # Cast real to tf.int32 to match predictions' dtype
    real = tf.cast(real, dtype=tf.int32)

    # Compare predictions with real values
    accuracies = tf.math.equal(real, predictions)

    # Create a mask to ignore padding tokens
    mask = tf.math.logical_not(tf.math.equal(real, PAD_ID))

    # Convert boolean tensors to float32 for calculations
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # Apply the mask and calculate the accuracy over non-padding tokens
    masked_accuracies = accuracies * mask

    # Handle cases where there are no non-padding tokens to avoid division by zero
    if tf.reduce_sum(mask) == 0:
        return tf.constant(0.0)

    return tf.reduce_sum(masked_accuracies) / tf.reduce_sum(mask)
