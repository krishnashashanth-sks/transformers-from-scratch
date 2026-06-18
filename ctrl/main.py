import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BasicCTRLModel
import torch.optim as optim
from dataset import dummy_dataloader
from train import train
from inference import generate_text

# Hyperparameters (example values, these would be tuned)
VOCAB_SIZE = 50257  # Example, e.g., for GPT-2 BPE tokenizer
MAX_SEQ_LEN = 1024 # Maximum sequence length
D_MODEL = 768      # Dimension of the model (embedding size)
N_HEADS = 12       # Number of attention heads
D_FF = 3072        # Dimension of the feed-forward network (usually 4 * D_MODEL)
N_LAYERS = 12      # Number of transformer layers
DROPOUT_RATE = 0.1

# Instantiate the model
model = BasicCTRLModel(
    VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, N_HEADS, D_FF, N_LAYERS, DROPOUT_RATE
)
print("Basic CTRL Model instantiated.")
print(model)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-1) # -1 is a common ignore index for padded tokens
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")

num_epochs = 3 # For demonstration, a small number of epochs

model.train() # Set the model to training mode

train(num_epochs,dummy_dataloader,model,optimizer,criterion,VOCAB_SIZE,device)

# Let's define some arbitrary token IDs for control codes and EOS
control_code_horror_id = torch.tensor([[1]]).to(device) # Example control code ID
control_code_comedy_id = torch.tensor([[2]]).to(device) # Another example control code ID
EOS_TOKEN = VOCAB_SIZE - 1 # Using the last ID in the vocabulary as EOS token

# Example 1: Generate text conditioned on 'horror' control code
print("\n--- Generating with 'Horror' control code ---")
# Dummy prompt tokens (e.g., conceptually: "The old house stood")
# Generate random token IDs for the prompt, ensuring they are not control code IDs or EOS.
# We use VOCAB_SIZE-1 as the exclusive upper bound for randint to avoid generating EOS within the prompt.
dummy_prompt_horror = torch.randint(max(control_code_horror_id.item(), control_code_comedy_id.item()) + 1, VOCAB_SIZE - 1, (1, 5)).to(device)
print(f"Initial prompt (token IDs): {dummy_prompt_horror.squeeze().tolist()}")

generated_ids_horror = generate_text(
    model,
    control_code_horror_id,
    dummy_prompt_horror,
    max_new_tokens=20, # Generate up to 20 new tokens
    device=device,
    eos_token_id=EOS_TOKEN
)
print(f"Generated sequence (token IDs): {generated_ids_horror}")
print(f"Sequence length: {len(generated_ids_horror)}")


# Example 2: Generate text conditioned on 'comedy' control code
print("\n--- Generating with 'Comedy' control code ---")
# Dummy prompt tokens (e.g., conceptually: "The cat jumped")
dummy_prompt_comedy = torch.randint(max(control_code_horror_id.item(), control_code_comedy_id.item()) + 1, VOCAB_SIZE - 1, (1, 4)).to(device)
print(f"Initial prompt (token IDs): {dummy_prompt_comedy.squeeze().tolist()}")

generated_ids_comedy = generate_text(
    model,
    control_code_comedy_id,
    dummy_prompt_comedy,
    max_new_tokens=20, # Generate up to 20 new tokens
    device=device,
    eos_token_id=EOS_TOKEN
)
print(f"Generated sequence (token IDs): {generated_ids_comedy}")
print(f"Sequence length: {len(generated_ids_comedy)}")

print("\nNote: Since we are using dummy token IDs and the model was trained on random data,")
print("the generated sequences are random numbers and do not form coherent text.")
print("In a real scenario, these token IDs would be converted back to words using a tokenizer.")