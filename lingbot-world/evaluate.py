import tensorflow as tf
from utils import create_padding_mask,create_look_ahead_mask,acc
from losses import loss_function
from accuracy import accuracy_function

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

@tf.function
def val_step(inp, tar,transformer):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _ = transformer(
        inp,
        tar_inp,
        training=False,
        enc_padding_mask=create_padding_mask(inp),
        look_ahead_mask=create_look_ahead_mask(tf.shape(tar_inp)[1]),
        dec_padding_mask=create_padding_mask(inp)
    )
    loss = loss_function(tar_real, predictions)

    val_loss(loss)
    val_accuracy(accuracy_function(tar_real, predictions))

print("Validation step function defined.")