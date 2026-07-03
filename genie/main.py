import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from model import GenieArchitecture

# Load the IMDB movie reviews dataset
# The 'with_info=True' argument allows us to get dataset information,
# which can be useful for understanding its structure.
(ds_train, ds_test), ds_info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True, # Returns (text, label) pairs
    with_info=True
)
# Configuration for tokenization and padding
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 256

# Create a TextVectorization layer for tokenization and vocabulary building
text_vectorization_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

# Adapt the TextVectorization layer to the training dataset to build the vocabulary
# We only adapt on the text, not the labels
train_text = ds_train.map(lambda text, label: text)
text_vectorization_layer.adapt(train_text)

# Get the vocabulary
vocab = text_vectorization_layer.get_vocabulary()
print(f"Vocabulary size: {len(vocab)}")
print(f"Top 10 words in vocabulary: {vocab[:10]}")

# Function to preprocess the dataset (tokenize and convert to IDs)
def preprocess_text_data(text, label):
    text = tf.strings.lower(text) # Convert text to lowercase
    text = tf.strings.regex_replace(text, b'<br />', b' ') # Remove HTML line breaks
    text = tf.strings.regex_replace(text, b'[^a-z ]', b'') # Remove non-alphabetic characters
    text = tf.strings.strip(text) # Remove leading/trailing whitespace

    # Apply TextVectorization
    text = text_vectorization_layer(text)
    return text, label

# Apply preprocessing to both training and test datasets
ds_train_processed = ds_train.map(preprocess_text_data)
ds_test_processed = ds_test.map(preprocess_text_data)

BUFFER_SIZE_PREFETCH = tf.data.AUTOTUNE
BATCH_SIZE = 64

# Determine the number of training examples from ds_info
num_train_examples = ds_info.splits['train'].num_examples

# Calculate sizes for train and validation splits
# Using 80% for training and 20% for validation from the original training set
train_size = int(0.8 * num_train_examples)
val_size = num_train_examples - train_size

print(f"Total training examples: {num_train_examples}")
print(f"Training set size: {train_size}")
print(f"Validation set size: {val_size}")

# Shuffle the training dataset and split into new training and validation sets
# Use num_train_examples as buffer_size to ensure proper shuffling of the entire dataset
ds_train_shuffled = ds_train_processed.shuffle(num_train_examples)
ds_val_processed = ds_train_shuffled.take(val_size)
ds_train_processed = ds_train_shuffled.skip(val_size)

# Apply batching and prefetching for all datasets
# The original ds_test_processed is already separated as the test set.

ds_train_final = ds_train_processed.batch(BATCH_SIZE).prefetch(BUFFER_SIZE_PREFETCH)
ds_val_final = ds_val_processed.batch(BATCH_SIZE).prefetch(BUFFER_SIZE_PREFETCH)
ds_test_final = ds_test_processed.batch(BATCH_SIZE).prefetch(BUFFER_SIZE_PREFETCH)

print(f"\nBatched training dataset: {ds_train_final}")
print(f"Batched validation dataset: {ds_val_final}")
print(f"Batched test dataset: {ds_test_final}")
print("Dataset splitting, batching, and prefetching complete.")

embedding_dim=128
conv_filters=64
conv_kernel_size=5
lstm_units=128

# Instantiate the GenieArchitecture model
model = GenieArchitecture(
    vocab_size=VOCAB_SIZE,
    embedding_dim=embedding_dim,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    conv_filters=conv_filters,
    conv_kernel_size=conv_kernel_size,
    lstm_units=lstm_units
)

# Build the model by calling model.build() with an appropriate input shape
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))

# Print the model summary
model.summary()
print("GenieArchitecture class typo fixed and model summary printed.")
# This ensures the model is compiled before calling .fit()

# 1. Choose a suitable loss function for binary classification
# Since the model uses a sigmoid activation for binary classification, BinaryCrossentropy is appropriate.
loss_fn = tf.keras.losses.BinaryCrossentropy()

learning_rate = 1e-4

# 2. Select an appropriate optimizer (e.g., Adam)
# learning_rate is available from previous cells
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 3. Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

print("Model re-compiled successfully with Adam optimizer and BinaryCrossentropy loss.")

EPOCHS = 10

# Train the model
history = model.fit(
    ds_train_final,
    epochs=EPOCHS,
    validation_data=ds_val_final
)

print(f"Model training complete after {EPOCHS} epochs.")
# Evaluate the model on the test dataset
print("\nEvaluating model on the test dataset...")
loss, accuracy = model.evaluate(ds_test_final)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Visualize training history
# history is available from the previous training step
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

print("Model evaluation and visualization of training history complete.")
# Example new movie reviews
new_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible film, a complete waste of time. I wish I hadn't watched it.",
    "It was okay, nothing special. I've seen better.",
    "A truly captivating story with brilliant performances. Highly recommend!"
]

# Preprocess the new reviews using the same text_vectorization_layer
# We need to ensure the preprocessing function handles a list of strings if directly applied, or iterate.
# For simplicity, let's process them one by one or vectorize the list.
processed_new_reviews = text_vectorization_layer(tf.constant(new_reviews))

# Make predictions
predictions = model.predict(processed_new_reviews)

# Interpret predictions
print("\nSentiment Predictions for New Reviews:")
for i, prediction in enumerate(predictions):
    sentiment = "positive" if prediction[0] > 0.5 else "negative"
    print(f"Review: \"{new_reviews[i][:50]}...\"\n  Probability of positive sentiment: {prediction[0]:.4f} ({sentiment})\n")
