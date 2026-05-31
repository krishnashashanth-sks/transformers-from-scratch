import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleGPT
from dataset import dataloader,char_to_idx,idx_to_char,vocab_size_char
from train import train
from generate import generate_text

embed_dim = 128
max_seq_len = 10
num_heads = 4
num_layers = 2
hidden_dim = embed_dim * 4
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the learning rate for the optimizer
learning_rate = 0.001
print(f"Learning rate: {learning_rate}")

# 2. Define the total number of training epochs
num_epochs = 50
print(f"Number of epochs: {num_epochs}")

# 3. Re-instantiate the SimpleGPT model with the character-level vocabulary size
gpt_model = SimpleGPT(vocab_size_char, embed_dim, max_seq_len, num_heads, num_layers, hidden_dim)
print(f"SimpleGPT model re-instantiated with vocab_size_char: {vocab_size_char}")

# 4. Move the gpt_model to the appropriate device
gpt_model.to(device)
print(f"Model moved to device: {device}")

# 5. Define the loss function
criterion = nn.CrossEntropyLoss()
print("CrossEntropyLoss criterion defined.")

# 6. Define the optimizer
optimizer = optim.Adam(gpt_model.parameters(), lr=learning_rate)
print(f"Adam optimizer defined with learning rate: {learning_rate}")

# 7.Training 
train_losses=train(gpt_model=gpt_model,dataloader=dataloader,optimizer=optimizer,criterion=criterion,num_epochs=num_epochs,device=device)

import matplotlib.pyplot as plt

# Plot the training losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title("GPT Model Training Loss")
plt.xlabel("Training Steps (Batches)")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

start_prompt = "ROMEO:"
num_chars_to_generate = 500
generation_temperature = 0.8 # Experiment with this value

print(f"Generating text with prompt: '{start_prompt}'")
generated_output = generate_text(
    gpt_model=gpt_model,
    start_string=start_prompt,
    num_generate=num_chars_to_generate,
    temperature=generation_temperature,
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    max_seq_len=max_seq_len,
    device=device
)

print("\n--- Generated Text ---")
print(generated_output)
print("----------------------")
