from model import MistralLarge2
from inference import generate_text
from tokenier import DummyTokenizer
import torch.optim as optim
import torch.nn as nn
import torch

# --- Dummy Training Loop Example ---

vocab_size = 1000
num_layers = 2
hidden_size = 256
num_heads = 8
num_kv_heads = 2
intermediate_size = 512
max_seq_len = 128
window_size = 7

model = MistralLarge2(
    vocab_size=vocab_size,
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    intermediate_size=intermediate_size,
    max_seq_len=max_seq_len,
    window_size=window_size
) 

# Define Loss Function (for language modeling, CrossEntropyLoss is common)
loss_fn = nn.CrossEntropyLoss()

# Define Optimizer (AdamW is a popular choice for Transformers)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Create a single batch of dummy data for demonstration
batch_size = 2
seq_len = 10

dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
# For language modeling, targets are typically the next tokens
dummy_target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"\nDummy input_ids shape: {dummy_input_ids.shape}")
print(f"Dummy target_ids shape: {dummy_target_ids.shape}")

# Training loop structure (single iteration for demonstration)
num_epochs = 100 # Just one epoch for this example

print("\nStarting a dummy training loop for 1 epoch...")

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    optimizer.zero_grad() # Zero gradients

    # Forward pass
    output = model(dummy_input_ids)

    # Reshape for CrossEntropyLoss: (N, C, ...) where C is vocab_size
    # output: (batch_size, seq_len, vocab_size)
    # target: (batch_size, seq_len)
    loss = loss_fn(output.view(-1, vocab_size), dummy_target_ids.view(-1))

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Dummy training loop completed.")

dummy_tokenizer = DummyTokenizer(vocab_size=vocab_size)

# Example prompt (using dummy token IDs for now)
# In a real scenario, you'd encode a string prompt: `tokenizer.encode("The quick brown fox")`
prompt_text = "The quick brown fox jumps over the lazy dog"

# For simplicity, we'll convert this to dummy input IDs directly
initial_input_ids = torch.tensor([dummy_tokenizer.encode(prompt_text)], dtype=torch.long)

print(f"Initial prompt (token IDs): {initial_input_ids.tolist()}")

# Generate text
generated_sequence = generate_text(model, dummy_tokenizer, initial_input_ids,vocab_size, max_new_tokens=20, temperature=0.7)

print("\nGenerated Sequence (token IDs):")
print(generated_sequence)
print("\nNote: The output consists of dummy token IDs as a real tokenizer and meaningful training data are not available in this setup.")