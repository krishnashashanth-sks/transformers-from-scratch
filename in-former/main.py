import tensorflow as tf
import numpy as np
import pandas as pd
from model import Informer
from data_generator import InformerDataGenerator
from evaluate import val_step
from train import train_step

# ==============================================================================
#  Data Preparation
# ==============================================================================

# 1. Define parameters for data preparation
seq_len = 96  # Encoder sequence length (e.g., history to look back)
label_len = 48 # Length of known history for the decoder before prediction
pred_len = 24 # Prediction horizon

# 2. Generate a synthetic time-series dataset (e.g., a sine wave with some noise)
n_samples = 1000
time = np.arange(n_samples)
sine_wave = np.sin(time / 10.0) * 10
noise = np.random.normal(0, 1, n_samples)
synthetic_data = sine_wave + noise
synthetic_data_reshaped = synthetic_data.reshape(-1, 1)

# 3. Generate dummy temporal features
dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_samples, freq='h'))
dummy_time_features = np.stack([
    dates.minute.values,
    dates.hour.values,
    dates.day.values,
    dates.dayofweek.values,
    dates.month.values,
    dates.year.values - dates.year.min() # Normalize year to start from 0
], axis=-1)

# ==============================================================================
#  Model Instantiation and Compilation
# ==============================================================================

# Determine feature dimensions
enc_in_dim = synthetic_data_reshaped.shape[-1]
dec_in_dim = synthetic_data_reshaped.shape[-1]
c_out_dim = synthetic_data_reshaped.shape[-1]

# Instantiate the Informer model
informer_model = Informer(
    enc_seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    enc_in=enc_in_dim,
    dec_in=dec_in_dim,
    c_out=c_out_dim,
    d_model=512,
    num_heads=8,
    e_layers=3,
    d_layers=2,
    d_ff=2048,
    dropout_rate=0.1,
    attn_factor=5,
    distil=True,
    activation='gelu'
)

# Compile the model
informer_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

print("Informer model instantiated and compiled successfully.")

# ==============================================================================
# Main Training Loop
# ==============================================================================

epochs = 10
batch_size = 32

total_samples = synthetic_data_reshaped.shape[0]
max_start_idx_overall = total_samples - (seq_len + pred_len)

train_split_ratio = 0.8
train_end_idx = int(max_start_idx_overall * train_split_ratio)

# Create training generator
train_generator = InformerDataGenerator(
    data=synthetic_data_reshaped,
    time_features=dummy_time_features,
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    batch_size=batch_size,
    shuffle=True
)
train_generator.indices = np.arange(train_end_idx + 1)
train_generator.on_epoch_end()

# Create validation generator
val_generator = InformerDataGenerator(
    data=synthetic_data_reshaped,
    time_features=dummy_time_features,
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    batch_size=batch_size,
    shuffle=False
)
val_generator.indices = np.arange(train_end_idx + 1, max_start_idx_overall + 1)

print(f"Training with {len(train_generator)} batches.")
print(f"Validating with {len(val_generator)} batches.\n")

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
val_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='val_mae')

print(f"Number of trainable variables in Informer model: {len(informer_model.trainable_variables)}\n")

for epoch in range(epochs):
    train_loss_metric.reset_state()
    train_mae_metric.reset_state()
    val_loss_metric.reset_state()
    val_mae_metric.reset_state()

    print(f"Epoch {epoch + 1}/{epochs}:")
    for batch_idx in range(len(train_generator)):
        inputs_batch, targets_batch = train_generator[batch_idx]
        train_step(
            informer_model,
            inputs_batch,
            targets_batch,
            informer_model.optimizer,
            informer_model.loss,
            train_loss_metric,
            train_mae_metric
        )

    for batch_idx in range(len(val_generator)):
        inputs_batch, targets_batch = val_generator[batch_idx]
        val_step(
            informer_model,
            inputs_batch,
            targets_batch,
            informer_model.loss,
            val_loss_metric,
            val_mae_metric
        )

    print(f"  Train Loss: {train_loss_metric.result():.4f}, Train MAE: {train_mae_metric.result():.4f}")
    print(f"  Val Loss: {val_loss_metric.result():.4f}, Val MAE: {val_mae_metric.result():.4f}")

    train_generator.on_epoch_end()

print("\nTraining complete!")


# ==============================================================================
# 13. Inference Example
# ==============================================================================
print("\n=======================")
print("Inference Example:")
print("=======================")

# Get a single batch from the validation generator for inference
# It's important to use unseen data if possible, but for demonstration, we use validation data
inputs_for_inference, true_targets_for_inference = val_generator[0]

# Make a prediction with the trained model
inference_predictions = informer_model(inputs_for_inference, training=False)

# Print shapes and a sample of predictions vs. true targets
print(f"Input enc_input shape: {inputs_for_inference[0].shape}")
print(f"Input dec_input shape: {inputs_for_inference[1].shape}")
print(f"True targets shape: {true_targets_for_inference.shape}")
print(f"Inference predictions shape: {inference_predictions.shape}")

print("\nSample from first item in batch:")
print("  True Targets (first 5 values): ", true_targets_for_inference[0, :5, 0].numpy())
print("  Predictions (first 5 values):  ", inference_predictions[0, :5, 0].numpy())

# Calculate MAE for the inference batch
inference_mae = tf.keras.metrics.MeanAbsoluteError()
inference_mae.update_state(true_targets_for_inference, inference_predictions)
print(f"\nInference MAE for batch: {inference_mae.result().numpy():.4f}")
