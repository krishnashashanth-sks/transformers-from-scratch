import tensorflow as tf

# --- Training Function --- #
def train_step(model, optimizer, loss_object, metrics, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    metrics['train_loss'].update_state(loss)
    metrics['train_accuracy'].update_state(targets, predictions)