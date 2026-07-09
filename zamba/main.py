import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import evaluate_epoch
from utils import calculate_perplexity,group_texts
from model import Zamba
from train import train_epoch
from inference import generate_text

tokenizer = AutoTokenizer.from_pretrained('gpt2')

raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
max_sequence_length = 1024

def tokenize_function(examples):
    texts = [str(t) for t in examples["text"] if t is not None]
    return tokenizer(texts, truncation=True, max_length=max_sequence_length, return_attention_mask=True)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"]
)

block_size = max_sequence_length # Using the same as tokenizer's max length for consistency
batch_size = 4 # Adjust based on your GPU memory

processed_datasets = tokenized_datasets.map(
    [group_texts,block_size],
    batched=True,
    num_proc=4, # Use multiple processes for faster grouping
    remove_columns=tokenized_datasets["train"].column_names # Remove all original columns
)

# Set the format to PyTorch tensors
processed_datasets.set_format(type="torch", columns=["input_ids", "labels"])

# Create DataLoaders
train_dataloader = DataLoader(
    processed_datasets["train"],
    shuffle=True,
    batch_size=batch_size
)

eval_dataloader = DataLoader(
    processed_datasets["validation"],
    shuffle=False,
    batch_size=batch_size
)

test_dataloader = DataLoader(
    processed_datasets["test"],
    shuffle=False,
    batch_size=batch_size
)

print(f"Processed dataset structure after grouping: {processed_datasets}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of validation batches: {len(eval_dataloader)}")
print(f"Number of test batches: {len(test_dataloader)}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Get vocab_size from the tokenizer
vocab_size = tokenizer.vocab_size

# Model parameters (example values, adjust as needed)
d_model = 256 # Should align with tokenizer's embedding size
d_inner = d_model * 2 # Common practice for FFN in transformers/Mamba
d_state = 16
num_mamba_blocks = 6
attention_interval = 6 # Global attention every 6 Mamba blocks

# Instantiate the Zamba model
model = Zamba(
    d_model=d_model,
    d_inner=d_inner,
    d_state=d_state,
    num_mamba_blocks=num_mamba_blocks,
    attention_interval=attention_interval,
    vocab_size=vocab_size # For language modeling head
).to(device)

print("Zamba model instantiated successfully and moved to device.")

# Define the loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)

print(f"Loss function (CrossEntropyLoss) defined. ignore_index: {loss_fn.ignore_index}")

# Define the learning rate
learning_rate = 1e-4

# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Optimizer (AdamW) initialized with learning rate: {learning_rate}")
# Main Training Loop
epochs = 3 # Define the total number of epochs

train_losses = []
val_losses = []
best_val_loss = float('inf')

print("Starting main training loop...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Train
    avg_train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
    train_losses.append(avg_train_loss)
    train_ppl = calculate_perplexity(torch.tensor(avg_train_loss))

    # Evaluate
    avg_val_loss = evaluate_epoch(model, eval_dataloader, loss_fn, device)
    val_losses.append(avg_val_loss)
    val_ppl = calculate_perplexity(torch.tensor(avg_val_loss))

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train PPL = {train_ppl:.4f} | Val Loss = {avg_val_loss:.4f}, Val PPL = {val_ppl:.4f}")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_zamba_model.pt')
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")

print("Training complete.")

print("Evaluating model performance on the test dataset...")

# Load the best saved model state dictionary
model.load_state_dict(torch.load('best_zamba_model.pt'))
model.to(device)

# Evaluate on the test set
avg_test_loss = evaluate_epoch(model, test_dataloader, loss_fn, device)
test_ppl = calculate_perplexity(torch.tensor(avg_test_loss))

print(f"\nFinal Test Loss = {avg_test_loss:.4f}")
print(f"Final Test Perplexity = {test_ppl:.4f}")

# Analyze the results
print("\n--- Analysis ---")
print("Comparing test results with training and validation results:")
print(f"  Training Loss (last epoch): {train_losses[-1]:.4f}, Perplexity: {calculate_perplexity(torch.tensor(train_losses[-1])):.4f}")
print(f"  Validation Loss (best): {best_val_loss:.4f}, Perplexity: {calculate_perplexity(torch.tensor(best_val_loss)):.4f}")
print(f"  Test Loss: {avg_test_loss:.4f}, Perplexity: {test_ppl:.4f}")

if avg_test_loss < best_val_loss:
    print("The test loss is lower than the best validation loss, indicating good generalization.")
elif avg_test_loss > best_val_loss and avg_test_loss < train_losses[-1]:
    print("The test loss is slightly higher than validation but lower than training, suggesting reasonable generalization but potential for minor overfitting to the validation set.")
else:
    print("The test loss is significantly higher than validation, potentially indicating overfitting or issues with generalization to unseen data.")


prompt_text = "The quick brown fox"
generated_text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt_text,
    max_length=100, # Generate up to 100 new tokens
    temperature=0.8, # Control randomness
    top_k=50, # Consider only top 50 probable tokens
    device=device
)

print(f"\nPrompt: {prompt_text}")
print(f"Generated Text: {generated_text}")
