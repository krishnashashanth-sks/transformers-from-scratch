from sklearn.model_selection import train_test_split
from utils import tokenize_and_create_sequences,pad_sequences
from layers import Transformer
from scheduler import CustomSchedule
import tensorflow as tf
from inference import predict_sentence
from train import train_model

# Generate a synthetic dataset
dataset = [
    "command: move forward 5 units",
    "response: moving forward by 5 units",
    "observation: object detected at 10,20",
    "command: analyze object",
    "response: analyzing object",
    "observation: object type is cube, color red",
    "command: pick up red cube",
    "response: picking up red cube",
    "command: rotate left 90 degrees",
    "response: rotating left by 90 degrees",
    "observation: energy level low",
    "command: recharge",
    "response: initiating recharge sequence",
    "command: report status",
    "response: status report: all systems nominal, energy 20%",
    "command: explore area",
    "response: beginning exploration of current area",
    "observation: no new objects found",
    "command: standby",
    "response: entering standby mode",
    "observation: unexpected sound detected",
    "command: investigate sound source",
    "response: investigating sound source",
    "observation: sound source identified as local wildlife",
    "command: return to base",
    "response: returning to designated base location"
]
# 3. Create a unique character-level vocabulary
all_characters = set()
for sentence in dataset:
    for char in sentence:
        all_characters.add(char)

# Add special tokens
special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
for token in special_tokens:
    for char in token: # Add each character of the special token
        all_characters.add(char)

# It's more robust to treat special tokens as single units, not character-by-character.
# Let's re-think this. The architecture specifies 'character-level embeddings'.
# So, if <SOS> is a token, should its characters '<', 'S', 'O', 'S', '>' be in the vocab?
# Or should '<SOS>' itself be a single entry in the vocab?
# The description "character-level embeddings will be used to capture sub-word semantics" implies
# that each character including those forming special tokens should be in the vocab.
# However, typically <SOS>, <EOS> are treated as single tokens in the vocabulary.
# Given "character-level embeddings" AND "prepend <SOS> to target inputs for the decoder and append <EOS> to target outputs",
# it's best to include the individual characters that form these special tokens in the character vocabulary,
# but also have unique IDs for the special tokens themselves (e.g., '<SOS>' maps to one ID).
# This way, when we tokenize, <SOS> can be a single token, but its embedding will be handled by character-level processing later.
# For now, let's assume the character vocabulary is strictly for *characters* appearing in data, and
# special tokens like <SOS> and <EOS> will be treated as single entities during tokenization.
# Let's adjust the vocabulary creation to include only individual characters, and manage special tokens separately.

# Re-create all_characters for actual data characters
all_characters = set()
for sentence in dataset:
    for char in sentence:
        all_characters.add(char)

# Convert set to sorted list to maintain consistent ordering
vocabulary = sorted(list(all_characters))

# Add special tokens to the vocabulary as unique entries *if they are not characters*
# For character-level processing, it's about the characters present in the *corpus*.
# The instructions for the model say "character-level embeddings", but then "prepend <SOS> to target inputs".
# This means <SOS> needs to be a distinct 'token' at some level. Let's make the vocab consist of *all unique characters*, and then assign special *IDs* for <SOS>, <EOS>, <PAD>, <UNK> which might not correspond to single characters.

# Let's create a character-level vocabulary first, and then handle special tokens' indices.
# The special tokens themselves are not individual characters from the text in the same way 'a' or ' ' are.
# So, the vocabulary should be: special_tokens + unique_characters_from_data.

# Ensure consistent special token IDs at the beginning of the vocabulary
special_token_list = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

# Get all unique characters from the dataset, excluding any characters that form part of our special tokens if they are not meant to be in the general character vocab
# For simplicity, we'll just add the special token strings directly to the list if they are not single characters.
# Or, we can have a character vocab and separately handle special 'word-level' tokens.
# Given 'character-level embeddings', the vocabulary should contain individual characters.
# Let's assume the special tokens themselves (e.g., '<SOS>') will be assigned IDs *outside* the character range,
# or will be represented by their constituent characters plus some logic for interpretation.
# The prompt clearly says: "character-level embeddings... Each character in the vocabulary will be mapped to a dense vector."
# And then: "special tokens like start-of-sequence (`<SOS>`), end-of-sequence (`<EOS>`), padding (`<PAD>`), and unknown (`<UNK>`)."

# This implies the special tokens themselves are part of the vocabulary, not their constituent characters.
# Let's redefine the vocabulary to correctly handle this: unique characters + the special tokens as atomic units.

# Collect all unique characters from the dataset
text_chars = set(char for sentence in dataset for char in sentence)

# Combine with special tokens. If a special token (like '<') is also a text character, it will be handled.
# However, typically special tokens are treated as single semantic units that map to a single ID.
# Let's create a combined vocabulary of actual characters and the special token strings.

vocab_list = sorted(list(text_chars)) # Ensure characters have consistent IDs
vocab_list = special_token_list + vocab_list # Add special tokens at the beginning

# Create mappings
char_to_id = {char: i for i, char in enumerate(vocab_list)}
id_to_char = {i: char for i, char in enumerate(vocab_list)}

# Store special token IDs
PAD_ID = char_to_id['<PAD>']
SOS_ID = char_to_id['<SOS>']
EOS_ID = char_to_id['<EOS>']
UNK_ID = char_to_id['<UNK>']

vocabulary_size = len(vocab_list)
input_seqs, target_seqs = tokenize_and_create_sequences(dataset, char_to_id, SOS_ID, EOS_ID, UNK_ID)

max_input_len = max(len(seq) for seq in input_seqs)
max_target_len = max(len(seq) for seq in target_seqs)

# Pad sequences
padded_input_seqs = pad_sequences(input_seqs, max_input_len, PAD_ID)
padded_target_seqs = pad_sequences(target_seqs, max_target_len, PAD_ID)

# Combine input and target sequences for consistent splitting
# Note: padded_target_seqs includes SOS and EOS for decoder input/output
X = padded_input_seqs
Y = padded_target_seqs

# First, split into training + validation and test sets
X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42 # 15% for testing
)

# Then, split the training + validation set into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.176, random_state=42 # Approximately 15% of original for validation (0.176 * 0.85 approx 0.15)
)

BATCH_SIZE = 2 # Given the small dataset, a small batch size is appropriate

# Create TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE)

d_model = 512
num_heads = 8
dff = 2048 # d_model * 4
num_layers = 6
dropout_rate = 0.1

# Vocabulary sizes (computed in a previous step)
input_vocab_size = vocabulary_size
target_vocab_size = vocabulary_size

# Positional encoding lengths (computed in a previous step)
pe_input = max_input_len
pe_target = max_target_len

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    pe_input=pe_input,
    pe_target=pe_target,
    rate=dropout_rate
)

print("Transformer model instantiated successfully.")
# Instantiate the custom learning rate schedule
learning_rate = CustomSchedule(d_model=d_model, warmup_steps=4000)

# Initialize the Adam optimizer with the custom learning rate schedule and recommended parameters
optimizer = tf.keras.optimizers.Adam(
    learning_rate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9
)
# Define the number of epochs and batch size
EPOCHS = 20

BATCH_SIZE = 2 # Given the small dataset, a small batch size is appropriate
train_model(EPOCHS,train_dataset,val_dataset)
# Create TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE)

# Example usage:
input_text = "command: analyze object"
predicted_response = predict_sentence(input_text, transformer, char_to_id, id_to_char, SOS_ID, EOS_ID, PAD_ID, max_input_len, max_target_len)

print(f"Input: {input_text}")
print(f"Predicted Response: {predicted_response}")

input_text_2 = "observation: object type is cube, color red"
predicted_response_2 = predict_sentence(input_text_2, transformer, char_to_id, id_to_char, SOS_ID, EOS_ID, PAD_ID, max_input_len, max_target_len)

print(f"\nInput: {input_text_2}")
print(f"Predicted Response: {predicted_response_2}")