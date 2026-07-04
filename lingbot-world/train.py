import time
import tensorflow as tf
from utils import create_look_ahead_mask,create_padding_mask
from losses import loss_function
from accuracy import accuracy_function
from evaluate import val_accuracy,val_loss,val_step

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

@tf.function
def train_step(inp, tar,transformer,optimizer):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp,
            tar_inp,
            training=True,
            enc_padding_mask=create_padding_mask(inp),
            look_ahead_mask=create_look_ahead_mask(tf.shape(tar_inp)[1]),
            dec_padding_mask=create_padding_mask(inp)
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))
def train_model(epochs,train_dataset,val_dataset):
    print("Starting training...")

    for epoch in range(epochs):
        start = time.time()
        # Reset metrics for the new epoch
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        # Training loop
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 1 == 0:
                print(f'Epoch {epoch + 1} Batch {batch}: Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        # Validation loop
        for (batch, (inp, tar)) in enumerate(val_dataset):
            val_step(inp, tar)

        print(f'Epoch {epoch + 1} Final Train Loss {train_loss.result():.4f} Train Accuracy {train_accuracy.result():.4f}')
        print(f'Epoch {epoch + 1} Final Val Loss {val_loss.result():.4f} Val Accuracy {val_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    print("Training complete.")