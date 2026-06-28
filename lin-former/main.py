from model import Linformer
from dataset import VOCAB_SIZE,x_train_padded,y_train,x_test_padded,y_test,SEQUENCE_LENGTH
import tensorflow as tf

# Model Parameters
vocab_size = VOCAB_SIZE  # Example vocabulary size
sequence_length = SEQUENCE_LENGTH # Max sequence length
d_model = 256       # Dimension of the model's embeddings and hidden states
k_dim = 64          # Reduced dimension for keys (k << sequence_length)
v_dim = 64          # Reduced dimension for values (v << sequence_length)
num_heads = 4       # Number of attention heads
ff_dim = 1024       # Dimension of the feed-forward network
num_blocks = 2      # Number of Linformer Encoder Blocks
num_classes = 10    # Number of output classes for classification
imdb_num_classes = 2 # IMDb is binary classification

# Create an instance of the Linformer model
linformer_model = Linformer(
    vocab_size=vocab_size,
    sequence_length=sequence_length,
    d_model=d_model,
    k_dim=k_dim,
    v_dim=v_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_blocks=num_blocks,
    num_classes=imdb_num_classes # IMDb is binary classification

)

# Build the model with a dummy input shape
linformer_model.build(input_shape=(None, sequence_length))

# Compile the model
linformer_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Linformer Model Summary:")
linformer_model.summary()

# Train the model (using a small number of epochs for demonstration)
epochs = 3
batch_size = 64

print(f"\nStarting training for {epochs} epochs...")
history = linformer_model.fit(
    x_train_padded,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2 # Use 20% of training data for validation
)

print("\nTraining complete! Evaluating on test data...")
loss, accuracy = linformer_model.evaluate(x_test_padded, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

print("\nPerforming inference on test data...")
predictions = linformer_model.predict(x_test_padded)
print(f"Predictions shape: {predictions.shape}")
# Optionally, print the first few predictions
print("First 5 predictions (probabilities per class):\n", predictions[:5])
# To get the predicted class labels (for a softmax output)
predicted_classes = tf.argmax(predictions, axis=1)
print("\nFirst 5 predicted class labels:\n", predicted_classes.numpy()[:5])