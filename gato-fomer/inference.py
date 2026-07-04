def predict_with_gato_transformer(model, input_token_ids):
  """
  Performs inference with the GatoLikeTransformer model.

  Args:
    model: The trained GatoLikeTransformer model.
    input_token_ids: A tf.Tensor or numpy array of token IDs, shape (batch_size, maxlen).

  Returns:
    The model's output logits or probabilities for the next token prediction.
  """
  predictions = model(input_token_ids, training=False)
  return predictions
