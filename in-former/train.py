import tensorflow as tf

@tf.function
def train_step(model, inputs, targets, optimizer, loss_object, train_loss_metric, train_mae_metric):
    enc_input, dec_input, enc_time_features, dec_time_features = inputs

    with tf.GradientTape() as tape:
        predictions = model(
            (enc_input, dec_input, enc_time_features, dec_time_features),
            training=True
        )
        loss = loss_object(targets, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_metric.update_state(loss)
    train_mae_metric.update_state(targets, predictions)

