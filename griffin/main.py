from evaluate import eval_step
from inference import inference
from model import GriffinModel
from train import train_step
import tensorflow as tf

# --- Usage Example --- #
print("\n--- Demonstrating Usage ---")

# Model parameters
num_blocks = 2
d_model = 128
num_heads = 4
dff = 256
target_vocab_size = 1000 # Example vocab size
seq_len = 20
batch_size = 8

# Instantiate the GriffinModel
model = GriffinModel(num_blocks, d_model, num_heads, dff, target_vocab_size)

# Dummy data for training/evaluation
dummy_inputs = tf.random.uniform((batch_size, seq_len), maxval=target_vocab_size, dtype=tf.int32)
dummy_targets = tf.random.uniform((batch_size, seq_len), maxval=target_vocab_size, dtype=tf.int32)

# Optimizer and Loss
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Metrics
metrics = {
    'train_loss': tf.keras.metrics.Mean(name='train_loss'),
    'train_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
    'val_loss': tf.keras.metrics.Mean(name='val_loss'),
    'val_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy'),
}

# --- Training Loop Example ---
print("\n--- Simulating Training ---")
epochs = 3
for epoch in range(epochs):
    metrics['train_loss'].reset_states()
    metrics['train_accuracy'].reset_states()

    train_step(model, optimizer, loss_object, metrics, dummy_inputs, dummy_targets)

    print(f"Epoch {epoch + 1}, "
          f"Loss: {metrics['train_loss'].result():.4f}, "
          f"Accuracy: {metrics['train_accuracy'].result():.4f}")

print("Training simulation complete.")

# --- Evaluation Example ---
print("\n--- Simulating Evaluation ---")
metrics['val_loss'].reset_states()
metrics['val_accuracy'].reset_states()

eval_step(model, loss_object, metrics, dummy_inputs, dummy_targets) # Using same dummy data for simplicity

print(f"Validation Loss: {metrics['val_loss'].result():.4f}, "
      f"Validation Accuracy: {metrics['val_accuracy'].result():.4f}")
print("Evaluation simulation complete.")

# --- Inference Example ---
print("\n--- Simulating Inference ---")
start_tokens = tf.constant([[1, 5, 20]], dtype=tf.int32) # Batch size 1, sequence length 3
inferred_sequence = inference(model, start_tokens, max_length=10)

print(f"Starting sequence: {start_tokens.numpy()}")
print(f"Inferred sequence (up to max_length=10): {inferred_sequence.numpy()}")
print("Inference simulation complete.")