
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tokenizers import Tokenizer
import torch
import os
from tokenizers import ByteLevelBPETokenizer, Tokenizer
import os
import networkx as nx
from datasketch import MinHashLSH
from datasketch import MinHash
from tqdm.auto import tqdm
import pandas as pd
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
import os
from utils import clean_text_initial,is_english,filter_low_quality_content,generate_shingles
from model import ClaudeLikeModel
from train import train_model

dataset_name = "EleutherAI/the_pile_openwebtext2_document"

# Load a small sample of the dataset for demonstration purposes
# In a full-scale project, you would load the entire dataset or stream it.
# For initial exploration, loading a small split or a few examples is sufficient.
try:
    print(f"Loading a sample from '{dataset_name}' dataset...")
    # Load a small sample, e.g., the first 1000 examples from the train split
    # This allows us to quickly inspect the data structure without downloading the full multi-TB dataset.
    # The actual 'the_pile_openwebtext2_document' is a part of The Pile, which is already quite large.
    # For real 'The Pile', one might need to stream or use a specific configuration/split if available
    # or download chunks manually.
    # Using `split='train[:1000]'` or similar is common for quick checks.
    # For this specific dataset, we might just load a small number of full documents if available
    # or use a smaller, more accessible dataset for demonstration if 'the_pile' is too large even for sample loading.

    # Let's try loading a small, manageable split or subset for demonstration.
    # If 'the_pile_openwebtext2_document' is still too large or requires credentials/specific setup,
    # we might default to a tiny dataset for illustration.
    # For now, let's attempt to load a very small subset of a publicly accessible dataset if 'the_pile' proves difficult for sampling.

    # A more practical approach for initial demonstration with 'the_pile' might be to load a very specific config or split.
    # Since the full 'the_pile' is massive, we'll demonstrate loading a small, readily available dataset
    # that mimics text data for the next steps.
    # We'll use 'wikitext-2-raw-v1' as an example for demonstration of text processing.
    # For actual 'The Pile', the process would be similar but involve significantly more resources.

    print("Loading 'wikitext-103-raw-v1' dataset as a proxy for a larger text corpus sample...")
    # Loading 5000 examples from wikitext-103, which is a larger dataset than wikitext-2
    raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train[:5000]')
    print(f"Sample data loaded. Number of entries: {len(raw_dataset)}")
    print("First entry of the raw dataset:")
    print(raw_dataset[0]['text'])

except Exception as e:
    print(f"Could not load the specified dataset sample. Error: {e}")
    print("This might be due to size, internet issues, or specific dataset configurations. "
          "Proceeding with an empty dataset for code structure demonstration.")
    raw_dataset = []

# Store the raw_dataset in a global variable for subsequent steps if it's not empty.
if raw_dataset:
    global df_raw_text
    df_raw_text = pd.DataFrame(raw_dataset)
    print("Raw dataset converted to DataFrame and stored as 'df_raw_text'.")
    print(df_raw_text.head())
else:
    print("No raw dataset loaded or converted to DataFrame.")

print("Applying initial cleaning to 'df_raw_text'...")
# Apply the cleaning function to the 'text' column
# Using .loc to avoid SettingWithCopyWarning, if df_raw_text was a slice
df_raw_text.loc[:, 'text'] = df_raw_text['text'].apply(clean_text_initial)

print("Initial cleaning complete. Calculating text lengths...")
df_raw_text['text_length'] = df_raw_text['text'].apply(len)

# Filter out documents shorter than a specified character count
min_char_count = 50 # Define a threshold for minimum character count
initial_rows = len(df_raw_text)
df_raw_text = df_raw_text[df_raw_text['text_length'] >= min_char_count]
rows_after_filter = len(df_raw_text)
print(f"Filtered out {initial_rows - rows_after_filter} documents shorter than {min_char_count} characters.")

print("Displaying first 5 rows of cleaned and filtered data:")
print(df_raw_text.head())
print("\nDescriptive statistics of text lengths after cleaning and filtering:")
print(df_raw_text['text_length'].describe())


print(f"Initial number of documents before language filtering: {len(df_raw_text)}")

# Apply the language detection function
# Using .loc to avoid SettingWithCopyWarning
df_raw_text.loc[:, 'is_english'] = df_raw_text['text'].apply(is_english)

# Filter the DataFrame to keep only English text
df_raw_text_filtered_lang = df_raw_text[df_raw_text['is_english'] == True].copy()

print(f"Number of documents after English language filtering: {len(df_raw_text_filtered_lang)}")

print("\nFirst 5 rows of the language-filtered DataFrame:")
print(df_raw_text_filtered_lang.head())

# Update df_raw_text to the filtered version for subsequent steps
df_raw_text = df_raw_text_filtered_lang

print(f"Initial number of documents before low-quality filtering: {len(df_raw_text)}")

# Apply the low-quality content filter
df_raw_text.loc[:, 'is_high_quality'] = df_raw_text['text'].apply(filter_low_quality_content)

# Filter the DataFrame to keep only high-quality text
df_raw_text_final_filtered = df_raw_text[df_raw_text['is_high_quality'] == True].copy()

print(f"Number of documents after low-quality filtering: {len(df_raw_text_final_filtered)}")

print("\nFirst 5 rows of the final filtered DataFrame:")
print(df_raw_text_final_filtered.head())

# Update df_raw_text to the final filtered version
df_raw_text = df_raw_text_final_filtered
print("\nDataFrame 'df_raw_text' updated with final filtered data.")

print("\nDescriptive statistics of text lengths after all cleaning and filtering:")
print(df_raw_text['text_length'].describe())

print("Generating MinHash signatures...")

# 1. Define number of permutations for MinHash
num_perm = 128 # A common choice for initial deduplication

# 2. Initialize an empty list to store MinHash signatures and their original indices
minhash_signatures = []

# 3. Iterate through each document in the df_raw_text DataFrame
# Use iterrows() to get both index and row data
for original_idx, row in tqdm(df_raw_text.iterrows(), total=len(df_raw_text), desc="Generating MinHash"): # Use original_idx for tracking
    text = row['text']

    # Generate shingles for the current text
    shingles = generate_shingles(text, k=3) # Using k=3 as defined previously

    # Create a MinHash object
    m = MinHash(num_perm=num_perm)

    # Update the MinHash object by adding each shingle
    if shingles:
        for shingle in shingles:
            m.update(str(shingle).encode('utf8')) # MinHash expects bytes

    # Append a tuple containing the original DataFrame index and the MinHash object
    minhash_signatures.append((original_idx, m))

print(f"Generated {len(minhash_signatures)} MinHash signatures.")
print(f"First 5 MinHash signature objects: {minhash_signatures[:5]}")

print("Initializing MinHash LSH and inserting signatures...")

# 1. Define LSH parameters
# The threshold controls the Jaccard similarity cutoff for candidate pairs.
# For num_perm=128, a threshold around 0.5-0.7 is often a good starting point.
threshold = 0.7 # Documents with Jaccard similarity >= this threshold are candidates

# 2. Initialize MinHashLSH
# The 'num_perm' for LSH should match the one used for MinHash signatures
lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

# 3. Insert all MinHash objects into the LSH index
# We store the original DataFrame index as the key in LSH
inserted_count = 0
for original_idx, m_hash in tqdm(minhash_signatures, desc="Inserting into LSH"):
    lsh.insert(original_idx, m_hash)
    inserted_count += 1

print(f"LSH index size: {inserted_count} signatures inserted.")

print("Finding candidate duplicate pairs using LSH...")

# Use a graph to represent connections between similar documents
# Each node is the original_idx of a document
# An edge means they are candidates for being duplicates
duplicate_graph = nx.Graph()

# Add all documents as nodes initially
for original_idx, _ in minhash_signatures:
    duplicate_graph.add_node(original_idx)

# Query LSH for each signature to find potential duplicates
for original_idx, m_hash in tqdm(minhash_signatures, desc="Querying LSH"): # Use tqdm here
    # Query LSH for items similar to m_hash. Returns a list of keys (original_idx)
    # that are candidates for being duplicates of m_hash.
    result = lsh.query(m_hash)

    # Add edges between the current document and its candidates
    for candidate_idx in result:
        if original_idx != candidate_idx: # A document is not a duplicate of itself
            duplicate_graph.add_edge(original_idx, candidate_idx)

print("Identifying duplicate groups...")
# Find connected components in the graph. Each component is a group of similar documents.
duplicate_groups = list(nx.connected_components(duplicate_graph))

# Determine which documents to keep and which to remove
# We'll keep the first document (smallest original_idx) in each connected component
# and remove the rest.
indices_to_keep = set()
indices_to_remove = set()

for group in duplicate_groups:
    # Convert set to list and sort to ensure consistent selection (e.g., smallest index)
    sorted_group = sorted(list(group))
    indices_to_keep.add(sorted_group[0]) # Keep the first one
    for idx in sorted_group[1:]:
        indices_to_remove.add(idx) # Remove the rest

print(f"Total documents: {len(df_raw_text)}")
print(f"Number of documents identified for removal: {len(indices_to_remove)}")

# Filter df_raw_text to keep only the selected documents
df_deduplicated = df_raw_text.loc[list(indices_to_keep)].copy()

print(f"Number of documents after deduplication: {len(df_deduplicated)}")
print("First 5 rows of the deduplicated DataFrame:")
print(df_deduplicated.head())

# Update df_raw_text to the deduplicated version for subsequent steps
df_raw_text = df_deduplicated
print("DataFrame 'df_raw_text' updated with deduplicated data.")

corpus = df_raw_text['text'].tolist()
print(f"Extracted {len(corpus)} documents for tokenizer training.")


# 2. Initialize a BPETokenizer
# ByteLevelBPETokenizer is often used for training from scratch
tokenizer = ByteLevelBPETokenizer()

# 3. Train the BPETokenizer on the list of texts
vocab_size = 5000 # Keep vocab_size small for demonstration with a small corpus
special_tokens = [
    "[UNK]", # Unknown token
    "[PAD]", # Padding token
    "[CLS]", # Classification token
    "[SEP]", # Separator token
    "[MASK]" # Mask token
]

print(f"Training tokenizer with vocab_size={vocab_size} and special_tokens={special_tokens}")
# The `train_from_iterator` method expects an iterator that yields strings
tokenizer.train_from_iterator(corpus, vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)

print("Tokenizer training complete.")

# 4. Save the trained tokenizer to a file
tokenizer_dir = "./tokenizer_data"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer_file_path = os.path.join(tokenizer_dir, "my_tokenizer.json")
tokenizer.save(tokenizer_file_path) # Save the tokenizer as a single JSON file
print(f"Tokenizer saved to {tokenizer_file_path}")

# 5. Load the saved tokenizer to confirm it works
# Fix: Use Tokenizer.from_file() to load the tokenizer from a single JSON file
loaded_tokenizer = Tokenizer.from_file(tokenizer_file_path)
print("Tokenizer loaded successfully.")

# Use the loaded tokenizer to encode a sample sentence and decode
sample_text = "This is a sample sentence for tokenization testing. It's quite interesting."
encoded = loaded_tokenizer.encode(sample_text)

print(f"\nSample Text: {sample_text}")
print(f"Encoded IDs: {encoded.ids}")
print(f"Tokens: {encoded.tokens}")
print(f"Decoded Text: {loaded_tokenizer.decode(encoded.ids)}")

# Verify special tokens are mapped
cls_id = loaded_tokenizer.token_to_id("[CLS]")
pad_id = loaded_tokenizer.token_to_id("[PAD]")
unk_id = loaded_tokenizer.token_to_id("[UNK]")

print(f"[CLS] token ID: {cls_id}")
print(f"[PAD] token ID: {pad_id}")
print(f"[UNK] token ID: {unk_id}")

# 1. Load the previously saved tokenizer
tokenizer_file_path = "./tokenizer_data/my_tokenizer.json"
if os.path.exists(tokenizer_file_path):
    loaded_tokenizer = Tokenizer.from_file(tokenizer_file_path)
    print("Tokenizer loaded successfully.")
else:
    print(f"Error: Tokenizer file not found at {tokenizer_file_path}")
    loaded_tokenizer = None

# 2. Define a maximum sequence length
# This should be consistent with the max_seq_len used in ClaudeLikeModel's initialization
max_seq_len = 512 # Example value, adjust as needed

# Configure tokenizer for truncation and padding
loaded_tokenizer.enable_truncation(max_length=max_seq_len)
loaded_tokenizer.enable_padding(direction='right', pad_id=loaded_tokenizer.token_to_id("[PAD]"), pad_type_id=0, pad_token="[PAD]", length=max_seq_len)

# Lists to store encoded input_ids and attention_masks
all_input_ids = []
all_attention_masks = []

print(f"Encoding and processing {len(df_raw_text)} documents...")
# 3. Iterate through the df_raw_text DataFrame and encode each text
for text in df_raw_text['text']:
    # a. Encode the text with add_special_tokens
    encoded_output = loaded_tokenizer.encode(text, add_special_tokens=True)

    # b. Extract input_ids and attention_mask
    all_input_ids.append(encoded_output.ids)
    all_attention_masks.append(encoded_output.attention_mask)

print("Encoding complete.")

# 4. Convert the lists to PyTorch tensors
input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long)

# 5. Print the shapes of the resulting tensors
print(f"Shape of input_ids_tensor: {input_ids_tensor.shape}")
print(f"Shape of attention_mask_tensor: {attention_mask_tensor.shape}")

# --- 2. Define training loop parameters ---
batch_size = 8 # Adjusted for demonstration with a small dataset and CPU
learning_rate = 5e-5
num_epochs = 3 # Small number for demonstration

# Ensure pad_token_id is available from the tokenizer (assuming it's 1 from previous output)
# If loaded_tokenizer is available from previous steps, use: loaded_tokenizer.token_to_id("[PAD]")
pad_token_id = 1 # Based on previous tokenizer output

print(f"Training parameters: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}, pad_token_id={pad_token_id}")

# --- 3. Create a torch.utils.data.TensorDataset ---

dataset = TensorDataset(input_ids_tensor, attention_mask_tensor)
print(f"TensorDataset created with {len(dataset)} samples.")


# --- 4. Create a torch.utils.data.DataLoader ---

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"DataLoader created with {len(dataloader)} batches.")

# --- 5. Instantiate the ClaudeLikeModel ---
# Re-using model parameters from previous model definition cell
vocab_size = 5000 # Updated to match tokenizer vocab_size
max_seq_len = 512 # Consistent with data preparation
embed_dim = 768
num_layers = 2 # Reduced for faster demonstration
num_heads = 12
num_kv_heads = 3
ffn_hidden_dim = 3072
dropout_rate = 0.1

model = ClaudeLikeModel(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embed_dim=embed_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    ffn_hidden_dim=ffn_hidden_dim,
    dropout_rate=dropout_rate
)
print("ClaudeLikeModel instantiated.")

# --- 6. Define the CrossEntropyLoss ---
# The labels for language modeling are the input_ids shifted by one position.
# We ignore the pad_token_id in the loss calculation.
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
print("CrossEntropyLoss defined, ignoring pad_token_id.")

# --- 7. Instantiate the AdamW optimizer ---
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
print("AdamW optimizer instantiated.")

# --- 8. Implement a learning rate scheduler ---
num_training_steps = len(dataloader) * num_epochs if dataloader else 0
num_warmup_steps = int(0.1 * num_training_steps) # 10% of total steps for warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
print(f"Learning rate scheduler initialized with {num_warmup_steps} warmup steps and {num_training_steps} total steps.")

# --- 9. Set up the device for training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")
num_epochs=100

train_model(num_epochs,dataloader,model,loss_fn,optimizer,scheduler,device)

print("Performing model inference...")

# Set the model to evaluation mode
model.eval()

# Get the pad_token_id from the tokenizer
pad_token_id = loaded_tokenizer.token_to_id("[PAD]")

# Define a sample input text
input_text = "Hello, how are you today?"

print(f"Input text for inference: '{input_text}'")

# Tokenize the input text
# Ensure add_special_tokens=True for consistency with training
encoded_input = loaded_tokenizer.encode(input_text, add_special_tokens=True)

# Get input_ids and attention_mask as tensors
input_ids_inference = torch.tensor(encoded_input.ids, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension
attention_mask_inference = torch.tensor(encoded_input.attention_mask, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension

print(f"Tokenized input IDs shape: {input_ids_inference.shape}")

# Perform inference
with torch.no_grad(): # Disable gradient calculation for inference
    # Generate logits for the next token
    # The model expects input_ids and an attention_mask
    logits = model(input_ids_inference, attention_mask=attention_mask_inference)

# Get the predicted next token (logits are for the last token in the sequence)
# The model output is (batch_size, seq_len, vocab_size)
# We're interested in the prediction for the last token in the input sequence
predicted_logits = logits[:, -1, :]
predicted_token_id = torch.argmax(predicted_logits, dim=-1).item()

# Decode the predicted token ID back to a word
predicted_token = loaded_tokenizer.decode([predicted_token_id])

print(f"Predicted next token ID: {predicted_token_id}")
print(f"Predicted next token: '{predicted_token}'")

# --- Example of generating a short sequence (simplified greedy decoding) ---
print("\nGenerating a short sequence (greedy decoding)...")

max_new_tokens = 10
generated_ids = input_ids_inference.tolist()[0] # Start with the input_ids

for _ in range(max_new_tokens):
    # Ensure the input to the model does not exceed max_seq_len
    # during generation. We take the last `max_seq_len` tokens if the sequence grows longer.
    input_to_model_list = generated_ids[-max_seq_len:]
    current_input_ids = torch.tensor(input_to_model_list, dtype=torch.long).unsqueeze(0).to(device)

    # The attention mask should match the length of current_input_ids
    current_attention_mask = torch.ones_like(current_input_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(current_input_ids, attention_mask=current_attention_mask)

    # Get the last token's logits for next prediction
    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).item()

    if next_token_id == pad_token_id: # Stop if padding token is generated
        break

    generated_ids.append(next_token_id)

full_generated_text = loaded_tokenizer.decode(generated_ids)
print(f"Generated sequence: '{full_generated_text}'")