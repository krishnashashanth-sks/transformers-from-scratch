import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import QwenModel
from train import train_model
import torch.nn as nn
import torch

# Example Usage (highly simplified parameters): Modified for lower RAM usage
vocab_size = 32000 # typical for Qwen
embed_dim = 256 # Dimension of embeddings and hidden states (Reduced from 1024)
num_layers = 2 # Number of transformer blocks (Reduced from 24)
num_heads = 4 # Number of attention heads (Reduced from 16)
ff_dim = embed_dim * 4 # Dimension of the feed-forward network
max_seq_len = 128 # Maximum sequence length (Reduced from 2048)

model = QwenModel(vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_len)
print(model)

# --- Training Setup (Conceptual) ---

# 1. Define Training Parameters
learning_rate = 1e-4
epochs = 1 # In reality, hundreds or thousands
batch_size = 2 # Very small for demonstration

# 2. Instantiate Optimizer and Loss Function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=-1) # -1 is a common ignore_index for padding

# 3. Create Dummy Data for Demonstration
# In a real scenario, this would come from a data loader processing text files.
# Let's create dummy sequences and corresponding target sequences
dummy_input_ids = torch.randint(0, vocab_size, (100, max_seq_len)) # 100 dummy sequences
dummy_target_ids = torch.randint(0, vocab_size, (100, max_seq_len)) # Targets for next token prediction
dummy_attention_mask = torch.ones((100, max_seq_len), dtype=torch.bool)

dummy_dataset = TensorDataset(dummy_input_ids, dummy_attention_mask, dummy_target_ids)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

print(f"Starting conceptual training for {epochs} epoch(s)...")

# 4. Training Loop
model.train() # Set model to training mode

train_model(epochs,dummy_dataloader,model,optimizer,criterion,vocab_size)

# Set the model to evaluation mode
model.eval()

# Create a new dummy input sequence for inference
# Let's use a batch size of 1 and the same max_seq_len
new_input_ids = torch.randint(0, vocab_size, (1, max_seq_len))
new_attention_mask = torch.ones((1, max_seq_len)).bool()

# Perform forward pass (inference)
with torch.no_grad(): # Disable gradient calculations for inference
  inference_logits = model(new_input_ids, attention_mask=new_attention_mask)

print(f"New input sequence shape: {new_input_ids.shape}")
print(f"Inference output (logits) shape: {inference_logits.shape}")
print("First 5 logits for the first token:")
print(inference_logits[0, 0, :5])
