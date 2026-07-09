# 1.Imports
import torch
import matplotlib.pyplot as plt
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from model import SambaModel
from train import train_model

# 2. Define parameters for the dummy dataset
vocab_size = 1000
sequence_length = 50
num_samples = 10000
batch_size = 32

print(f"Dataset Parameters: vocab_size={vocab_size}, sequence_length={sequence_length}, num_samples={num_samples}, batch_size={batch_size}")

# 3. Generate a dummy dataset of token IDs
# Input sequences: (num_samples, sequence_length)
# Values are random integers between 0 and vocab_size - 1
input_sequences_full = torch.randint(0, vocab_size, (num_samples, sequence_length), dtype=torch.long)

# For language modeling, target sequences are typically the input shifted by one token.
# The last token in the input sequence will have its target as a padding token (e.g., 0).
# Or, for simplicity in dummy data, we can just make the target the input itself for now
# and adjust later if needed for a specific loss function.
# Let's create target sequences as input sequences shifted by one for next token prediction.
target_sequences_full = torch.cat([
    input_sequences_full[:, 1:],
    torch.zeros(num_samples, 1, dtype=torch.long) # Pad last token target with 0 or a special EOS token
], dim=1)

# 4. Split the dummy dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    input_sequences_full,
    target_sequences_full,
    test_size=0.2, # 20% for validation
    random_state=42 # for reproducibility
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# 5. Create TensorDataset objects
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


# 6. Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

### Training
# 1. Set up the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define the model parameters
num_layers = 4 # Example: 4 Samba blocks
d_state = 16 # Example: d_state for MambaLayer
conv_kernel_size = 4 # Example: convolution kernel size for MambaLayer
d_inner_mamba = None # Optional, will use default in MambaLayer
d_inner_swiglu = None # Optional, will use default in SwiGLU
bias = False # Example: whether to use bias in linear layers
vocab_size = 1000
d_model = 128
num_heads = 8
window_size = 5

print(f"Model Parameters: num_layers={num_layers}, d_state={d_state}, conv_kernel_size={conv_kernel_size}, bias={bias}")

# 3. Instantiate the SambaModel

model = SambaModel(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    d_state=d_state,
    num_heads=num_heads,
    window_size=window_size,
    d_inner_mamba=d_inner_mamba,
    conv_kernel_size=conv_kernel_size,
    d_inner_swiglu=d_inner_swiglu,
    bias=bias
)

# 4. Move the initialized SambaModel to the selected device
model.to(device)
print(f"SambaModel moved to {device}.")

num_epochs = 5
learning_rate = 0.001

print(f"Training parameters: num_epochs={num_epochs}, learning_rate={learning_rate}")

# Call the train_model function
train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device
)

# Plotting the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

### Evaluating
# 1. Set the model to evaluation mode
model.eval()
print("Model set to evaluation mode.")

# 2. Create a dummy input sequence
# This should be a tensor of token IDs, similar to the training input
dummy_input_ids = torch.randint(0, vocab_size, (1, sequence_length), dtype=torch.long).to(device)
print(f"Dummy input shape: {dummy_input_ids.shape}")

# 3. Perform inference
with torch.no_grad(): # Disable gradient calculation for inference
    dummy_output = model(dummy_input_ids)

print(f"Dummy output shape: {dummy_output.shape}")

# 4. Interpret the output (e.g., get the predicted next token for each position)
# The output is (batch_size, sequence_length, vocab_size)
# To get the predicted token ID, we take the argmax along the vocab_size dimension
predicted_token_ids = torch.argmax(dummy_output, dim=-1)

print("Inference successful. Predicted token IDs:")
print(predicted_token_ids)

print("Inference on a dummy input completed.")