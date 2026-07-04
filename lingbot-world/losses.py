import tensorflow as tf
from main import PAD_ID

def loss_function(real, pred):
    # Create a SparseCategoricalCrossentropy object
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Calculate the loss for all tokens
    loss_ = loss_object(real, pred)

    # Create a mask to ignore padding tokens
    # PAD_ID was defined in a previous cell as the ID for the padding token
    mask = tf.math.logical_not(tf.math.equal(real, PAD_ID))
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss
    loss_ *= mask

    # Return the sum of the masked loss divided by the sum of the mask
    # This gives the average loss over non-padding tokens
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

print("Loss function defined.")