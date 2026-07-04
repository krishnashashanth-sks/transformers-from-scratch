import torch
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import datasets
from transformers import AutoTokenizer
from model import Granite4_0Model
from train import train_model
from evaluate import evaluate_model
from inference import generate_text
from losses import calculate_perplexity

# Create a dummy text file with corrected content to avoid empty entries
dummy_text_content = """This is a sample sentence for our dummy dataset.
Large language models are trained on vast amounts of text data.
Tokenization is a crucial step in natural language processing.
We are trying to implement Granite 4.0 architecture.
This is an additional sentence to ensure enough data for splits.
"""

# Reuse the existing dummy file path
dummy_file_path = "dummy_text_dataset.txt"

# Write the corrected content to the file
with open(dummy_file_path, "w") as f:
    f.write(dummy_text_content)

print(f"Dummy text file re-created with corrected content at: {dummy_file_path}")

# Re-load the local text file using datasets.load_dataset with the 'text' builder
print("Re-loading the dummy text dataset...")
dataset = datasets.load_dataset('text', data_files={'train': dummy_file_path})

print("Dataset re-loaded successfully:")
print(dataset)
print("First entry of the training split (after correction):")
print(dataset['train'][0])

max_sequence_length = 128 # Define a maximum sequence length for tokenization

# Choose a pre-trained tokenizer, e.g., 'gpt2'
print("Loading pre-trained GPT2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    # Tokenize the 'text' column, truncating to max_sequence_length
    return tokenizer(examples["text"], truncation=True, max_length=max_sequence_length)

print("Applying tokenization to the dataset...")
# Apply the tokenize_function to the entire dataset using map
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Dataset tokenized successfully.")
print(tokenized_dataset)

print("Splitting the tokenized dataset into train, validation, and test sets...")
# For a small dummy dataset (5 rows), we need to carefully manage splits
# First, split the single 'train' split into a temporary 'train_and_val' and 'test_dataset'
# test_size=1 means 1 example for the test set
split_temp_test = tokenized_dataset['train'].train_test_split(test_size=1, seed=42)

temp_train_val_dataset = split_temp_test['train'] # This will have 4 examples
test_dataset = split_temp_test['test'] # This will have 1 example

# Now, split the 'temp_train_val_dataset' into the final 'train_dataset' and 'validation_dataset'
# test_size=0.25 (1 example out of 4) for validation, ensuring 1 train and 1 validation example
split_train_val = temp_train_val_dataset.train_test_split(test_size=0.25, seed=42)

train_dataset = split_train_val['train'] # This will have 3 examples
validation_dataset = split_train_val['test'] # This will have 1 example

# Update the dataset dictionary with the new splits
dataset_splits = datasets.DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Define batch size
batch_size = 2 # Using a small batch size for the dummy dataset

# Initialize the DataCollator for Language Modeling
# For causal language modeling (like GPT2), we usually set mlm=False.
# The tokenizer's pad_token_id is crucial for padding.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Creating DataLoaders for train, validation, and test sets...")

# Create DataLoader instances for each split
# The collate_fn will handle padding and formatting for batches
train_dataloader = DataLoader(
    dataset_splits["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)

validation_dataloader = DataLoader(
    dataset_splits["validation"],
    shuffle=False, # No need to shuffle validation/test sets
    batch_size=batch_size,
    collate_fn=data_collator
)

test_dataloader = DataLoader(
    dataset_splits["test"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator
)

# 1. Define Model Hyperparameters
# These are generic placeholder values to be replaced with actual Granite 4.0 specifications once researched.
vocab_size = len(tokenizer)  # Use the tokenizer's vocabulary size
max_seq_len = max_sequence_length # Use the defined maximum sequence length
hidden_size = 768            # Dimension of the embedding and transformer layers
num_heads = 12               # Number of attention heads
num_layers = 6               # Number of transformer blocks
dropout_rate = 0.1           # Dropout probability
intermediate_size = 3072     # Dimension of the feed-forward network (often 4 * hidden_size)

print(f"Model Hyperparameters:\n"
      f"  vocab_size: {vocab_size}\n"
      f"  max_seq_len: {max_seq_len}\n"
      f"  hidden_size: {hidden_size}\n"
      f"  num_heads: {num_heads}\n"
      f"  num_layers: {num_layers}\n"
      f"  dropout_rate: {dropout_rate}\n"
      f"  intermediate_size: {intermediate_size}")
print("Instantiating Granite4_0Model...")

model = Granite4_0Model(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
    intermediate_size=intermediate_size
)

print("Model instantiated successfully.")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the re-instantiated model to the device
model.to(device)
print(f"Model moved to device: {device} (after re-instantiation)")

# Print the model structure
print("\nModel Architecture:")
print(model)

# Optional: Print total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")

# 1. Define Training Hyperparameters
learning_rate = 5e-5           # Learning rate for the optimizer
num_epochs = 3                 # Number of training epochs
warmup_steps = 500             # Number of steps for learning rate warm-up
gradient_accumulation_steps = 1 # Number of updates steps to accumulate before performing a backward/update pass
max_grad_norm = 1.0            # Maximum gradient norm for clipping
weight_decay = 0.01            # Weight decay for regularization

# 2. Initialize the loss function
# Use CrossEntropyLoss and explicitly ignore -100, which is typically used by DataCollatorForLanguageModeling for padding.
loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
print(f"Loss function initialized: {loss_function}")

# 3. Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print(f"Optimizer initialized: {optimizer}")

# 4. Set up a learning rate scheduler
# Calculate total training steps
num_update_steps_per_epoch = len(train_dataloader) / gradient_accumulation_steps
num_training_steps = int(num_epochs * num_update_steps_per_epoch)

# Create a linear warm-up and decay schedule
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)
print(f"Learning rate scheduler initialized. Total training steps: {num_training_steps}, Warmup steps: {warmup_steps}")

# 5. Define the device and move the model to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")

# 6. Print the configured hyperparameters and the device being used
print("\nConfigured Training Pipeline Hyperparameters:")
print(f"  Learning Rate: {learning_rate}")
print(f"  Number of Epochs: {num_epochs}")
print(f"  Warmup Steps: {warmup_steps}")
print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
print(f"  Max Gradient Norm: {max_grad_norm}")
print(f"  Weight Decay: {weight_decay}")
print(f"  Device: {device}")
train_model(num_epochs,train_dataloader,validation_dataloader,model,optimizer,lr_scheduler,vocab_size,max_grad_norm,loss_function,device)

total_test_loss,test_steps=evaluate_model(model,test_dataloader,loss_function,vocab_size,device)

avg_test_loss = total_test_loss / test_steps
avg_test_perplexity = calculate_perplexity(avg_test_loss)

print(f"\nTest Evaluation Results:")
print(f"  Average Test Loss: {avg_test_loss:.4f}")
print(f"  Average Test Perplexity: {avg_test_perplexity:.2f}")


# Define a sample prompt
prompt_text = "This is a test of the Granite 4.0 model architecture."

# Define generation parameters
max_gen_length = 50  # Maximum length of the generated sequence
# Using greedy decoding first, then can experiment with temperature, top_k, top_p
generated_text_greedy = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt_text,
    max_length=max_gen_length,
    device=device
)

print(f"\nGreedy Generated Text:\nPrompt: {prompt_text}\nGenerated: {generated_text_greedy}")

# Example with sampling (e.g., top-k or top-p)
generated_text_sampled = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt_text,
    max_length=max_gen_length,
    temperature=0.7, # Lower temperature for less randomness
    top_k=50,       # Consider top 50 tokens
    device=device
)

print(f"\nTop-K Sampled Generated Text:\nPrompt: {prompt_text}\nGenerated: {generated_text_sampled}")
