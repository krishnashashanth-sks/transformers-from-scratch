import torch.optim as optim
from torch.utils.data import DataLoader
from model import GLM5Model
from train import train_epoch
from evaluate import evaluate_model
from dataset import GLM5Dataset
from tokenizer import BasicTokenizer
import torch
import torch.nn as nn
from utils import generate_synthetic_text,preprocess_text
from inference import generate_text

vocab_size = 30000  # Example vocabulary size
embed_dim = 768     # Example embedding dimension
num_heads = 12      # Example number of attention heads
ff_dim = 3072       # Example feed-forward dimension
num_layers = 6      # Example number of transformer layers
max_len = 512       # Example maximum sequence length
max_seq_len=50

# Initialize the GLM5Model
glm5_model = GLM5Model(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len)

# Print the model architecture
print("\nGLM5 Model Architecture:")
print(glm5_model)

# Parameters for synthetic data generation
NUM_SAMPLES = 1000  # Number of synthetic sentences
MIN_WORDS_PER_SAMPLE = 10
MAX_WORDS_PER_SAMPLE = 50
UNIQUE_WORDS = 2000 # Number of unique words in our synthetic vocabulary

# Generate a list of unique synthetic words
synthetic_vocab = [f"word_{i}" for i in range(UNIQUE_WORDS)]


synthetic_data = generate_synthetic_text(NUM_SAMPLES, MIN_WORDS_PER_SAMPLE, MAX_WORDS_PER_SAMPLE, synthetic_vocab)

updated_synthetic_vocab = list(synthetic_vocab)
if '[UNK]' not in updated_synthetic_vocab:
    updated_synthetic_vocab.append('[UNK]')
if '[PAD]' not in updated_synthetic_vocab:
    updated_synthetic_vocab.append('[PAD]')

tokenizer = BasicTokenizer(updated_synthetic_vocab)

glm5_dataset = GLM5Dataset(synthetic_data, tokenizer, max_seq_len)

print(f"GLM5Dataset created with {len(glm5_dataset)} samples.")

BATCH_SIZE = 32 # Define the batch size for training

# Create the DataLoader
glm5_dataloader = DataLoader(glm5_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"DataLoader created with batch size: {BATCH_SIZE}")
print(f"Number of batches: {len(glm5_dataloader)}")

# Assuming glm5_model and tokenizer are already defined from previous steps

# 1. Define the loss function
# We use tokenizer.pad_id to ensure padding tokens do not contribute to the loss
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

# 2. Select an optimizer
# Initialize AdamW optimizer with model parameters and a learning rate
LEARNING_RATE = 1e-4 # Example learning rate
optimizer = optim.AdamW(glm5_model.parameters(), lr=LEARNING_RATE)

print(f"Loss function (CrossEntropyLoss) initialized with ignore_index={tokenizer.pad_id}.")
print(f"Optimizer (AdamW) initialized with learning rate={LEARNING_RATE}.")

NUM_EPOCHS = 3 # Define the number of training epochs

# Set up device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
glm5_model.to(device)

print(f"Training GLM5Model on device: {device}")

# Main training loop
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    avg_train_loss = train_epoch(glm5_model, glm5_dataloader, optimizer, loss_fn, device)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

print("Training complete.")

avg_eval_loss, perplexity = evaluate_model(glm5_model, glm5_dataloader, loss_fn, device)

sample_prompt = "word_1 word_2"
max_generation_len = 20 # Maximum number of tokens to generate

print(f"\nGenerating text with prompt: '{sample_prompt}'...")
generated_sequence = generate_text(glm5_model, tokenizer, sample_prompt, max_generation_len, device)

print(f"Generated sequence: {generated_sequence}")