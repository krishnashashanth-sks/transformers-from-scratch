# --- Evaluation Function --- #
def eval_step(model, loss_object, metrics, inputs, targets):
    predictions = model(inputs, training=False)
    loss = loss_object(targets, predictions)

    metrics['val_loss'].update_state(loss)
    metrics['val_accuracy'].update_state(targets, predictions)

