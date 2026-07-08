import tensorflow as tf

@tf.function
def val_step(model, inputs, targets, loss_object, val_loss_metric, val_mae_metric):
    enc_input, dec_input, enc_time_features, dec_time_features = inputs

    predictions = model(
        (enc_input, dec_input, enc_time_features, dec_time_features),
        training=False
    )
    loss = loss_object(targets, predictions)

    val_loss_metric.update_state(loss)
    val_mae_metric.update_state(targets, predictions)

