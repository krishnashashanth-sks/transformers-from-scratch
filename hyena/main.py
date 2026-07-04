import torch
from torch.utils.data import  DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from model import HyenaModel
from dataset import LanguageModelingDataset
from inference import generate_text
from utils import group_texts
import torch.nn as nn
from train import train_model

NUM_LAYERS = 4
DIM = 256
VOCAB_SIZE = 10000 # Example vocab size
SEQ_LEN = 512 # Example sequence length
ORDER = 2 # Number of convolutions/gating in Hyena operator
FILTER_LENGTH = 64 # Length of the kernel for Conv1d (conceptual)
DROPOUT = 0.1
BATCH_SIZE = 2

# --- 1. Load a simple text dataset (e.g., 'wikitext' for language modeling) ---
# Using 'wikitext-2-raw-v1' as it's small and suitable for demonstration.
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print("Dataset loaded.")

# --- 2. Initialize tokenizer ---
# Using a basic tokenizer. You might need a more advanced one for better performance.
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Tokenizer initialized.")

# --- 3. Preprocess the dataset ---
# We'll concatenate all text and then split it into fixed-size chunks.
# This is a common approach for language modeling with Transformer-like models.

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN)

# Tokenize all texts
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4, # Use multiple processes for faster tokenization
    remove_columns=["text"], # Remove the original text column
)

print(f"Grouping texts into blocks of {SEQ_LEN} tokens...")
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=4, # Use multiple processes for faster grouping
)
print("Texts grouped.")

# Prepare training and validation datasets and dataloaders
train_dataset = LanguageModelingDataset(lm_datasets["train"])
val_dataset = LanguageModelingDataset(lm_datasets["validation"])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of validation batches: {len(val_dataloader)}")

# Update VOCAB_SIZE based on the tokenizer's vocabulary size
VOCAB_SIZE = tokenizer.vocab_size
print(f"Updated VOCAB_SIZE based on tokenizer: {VOCAB_SIZE}")

# Verify a batch
for batch in train_dataloader:
    print("Sample batch input_ids shape:", batch["input_ids"].shape)
    print("Sample batch labels shape:", batch["labels"].shape)
    break
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Ensure the model parameters match the previously defined ones
model = HyenaModel(NUM_LAYERS, DIM, VOCAB_SIZE, ORDER, SEQ_LEN, FILTER_LENGTH, DROPOUT).to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # Ignore padding in loss

# Training loop parameters
EPOCHS = 1 # Keeping it short for demonstration
train_model(EPOCHS,train_dataloader,val_dataloader,model,optimizer,criterion,device)

print("\n--- Demonstrating Text Generation ---")

# Define a starting prompt
start_prompt = "The quick brown fox jumps over the lazy"

# Generate text
generated_output = generate_text(model, tokenizer, start_prompt, max_length=50, device=device, temperature=1.0, top_k=50)

print(f"Prompt: {start_prompt}")
print(f"Generated text: {generated_output}")