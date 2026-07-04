import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from utils import clean_text
from dataset import LanguageModelingDataset
from torch.utils.data import DataLoader
from model import GPTModel
import torch.nn as nn
from train import train_model
from inference import generate_text

print("Loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

print("Dataset loaded successfully.")
print(dataset)


print("Cleaning training data...")
# Apply cleaning to the 'train' split
dataset['train'] = dataset['train'].map(clean_text, batched=True, remove_columns=['text'])

print("Training data cleaned successfully.")
# Remove rows where 'text' might have become an empty list after cleaning
dataset['train'] = dataset['train'].filter(lambda x: len(x['text']) > 0)

print("Loading pre-trained GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set pad_token to eos_token to handle padding consistently for decoder-only models
# If the tokenizer does not have a pad_token, we set it to eos_token as a common practice for GPT-like models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully.")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Tokenizer pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

block_size = 128 # Define a reasonable block size for tokenization

def tokenize_function(examples):
    # Tokenize the texts. Truncation is important to ensure all examples fit into the block_size
    # and padding is set to 'max_length' to ensure consistent sequence lengths within a batch.
    # return_special_tokens_mask=True is useful for later masking during training if needed.
    return tokenizer(examples['text'], truncation=True, max_length=block_size, padding='max_length')

print(f"Tokenizing dataset with block size: {block_size}...")

# Apply tokenization to all splits of the dataset
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset['train'].column_names
)
# Create PyTorch Dataset instances for each split
train_dataset = LanguageModelingDataset(tokenized_datasets['train'])
val_dataset = LanguageModelingDataset(tokenized_datasets['validation'])
test_dataset = LanguageModelingDataset(tokenized_datasets['test'])

batch_size = 8 # Define a reasonable batch size

print(f"Creating PyTorch DataLoaders with batch size: {batch_size}...")
# Create PyTorch DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model (using values from previous steps or reasonable defaults)
vocab_size = len(tokenizer) # From tokenizer loaded previously
block_size = 128 # From previously defined block_size
embed_dim = 256 # Embedding dimension
num_heads = 4 # Number of attention heads
num_layers = 4 # Number of transformer layers
ff_dim = embed_dim * 4 # Feed-forward dimension (typically 4x embed_dim)
dropout_rate = 0.1

gpt_model = GPTModel(vocab_size, block_size, embed_dim, num_heads, num_layers, ff_dim, dropout_rate)

print(f"GPTModel instantiated with parameters:\n"
      f"  Vocab Size: {vocab_size}\n"
      f"  Block Size: {block_size}\n"
      f"  Embedding Dim: {embed_dim}\n"
      f"  Number of Heads: {num_heads}\n"
      f"  Number of Layers: {num_layers}\n"
      f"  Feed Forward Dim: {ff_dim}\n"
      f"  Dropout Rate: {dropout_rate}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
gpt_model.to(device)
print("Model moved to device.")

# Define hyperparameters
learning_rate = 3e-4
weight_decay = 0.01
num_epochs = 5# A reasonable number of epochs for initial training

# Define Loss Function
# CrossEntropyLoss is suitable for language modeling.
# Set ignore_index to tokenizer.pad_token_id so padded tokens don't contribute to the loss.
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
print(f"Loss function (CrossEntropyLoss) defined with ignore_index: {tokenizer.pad_token_id}")

# Define Optimizer
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print(f"Optimizer (AdamW) defined with learning rate: {learning_rate} and weight decay: {weight_decay}")

# Define Learning Rate Scheduler
# T_max is the total number of training iterations or epochs (here, epochs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
print(f"Learning rate scheduler (CosineAnnealingLR) defined with T_max: {num_epochs} epochs and eta_min: 1e-6")

print("Training environment configured.")
train_losses, val_losses, val_perplexities = train_model(
    gpt_model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, num_epochs, device, tokenizer
)
# Assuming train_losses, val_losses, val_perplexities are populated after training completes
# If the previous cell ran partially, you might need to re-run the train_model function
# or manually populate these lists for demonstration if the kernel was interrupted.

print("Generating plots for training and validation metrics...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_perplexities, label='Validation Perplexity', color='orange')
plt.title('Perplexity over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Plots generated successfully. The training and evaluation phase is complete.")

print("Generating text with a new prompt...")
new_prompt = "The early bird catches the"

# Ensure the model is on the correct device for generation
gpt_model.to(device)

generated_continuation_2 = generate_text(
    gpt_model, tokenizer, new_prompt, max_length=50, temperature=0.8, device=device
)

full_generated_text_2 = new_prompt + generated_continuation_2
print(f"\nPrompt: {new_prompt}")
print(f"Generated Text: {full_generated_text_2}")
