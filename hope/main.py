from dataloader import SimulatedDataLoader
from tokenizer import SimulatedTokenizer
from inference import generate_sequence
import torch
from evaluate import evaluate_model
from train import train_model
import torch.optim as optim
from model import HOPE_Architecture
import torch.nn as nn

vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 2
max_seq_len = 256
context_dim = 256
context_dim = 256 
seq_len = 10
batch_size = 4

hope_model = HOPE_Architecture(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=max_seq_len, context_dim=context_dim)

# Define Loss Function (for next token prediction, typical for language models)
loss_function = nn.CrossEntropyLoss()

# Define Optimizer for Slow Weights
learning_rate = 1e-4
optimizer = optim.AdamW(hope_model.parameters(), lr=learning_rate)

num_epochs = 5 # Number of training epochs
num_batches_per_epoch = 10 # Simulate a few batches per epoch

train_model(num_epochs,num_batches_per_epoch,vocab_size,batch_size,seq_len,optimizer,hope_model,loss_function)

# First, set model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hope_model.to(device)

# Initialize simulated data loader
num_eval_batches = 5
sim_eval_data_loader = SimulatedDataLoader(vocab_size, batch_size, seq_len, num_eval_batches)

# Run evaluation
eval_loss, eval_ppl, eval_acc = evaluate_model(hope_model, sim_eval_data_loader, vocab_size, device=device)

# Initialize simulated tokenizer
sim_tokenizer = SimulatedTokenizer(vocab_size)

# Simulate a prompt (batch_size=1, seq_len=some_length)
prompt_text = "This is a simulated prompt for the HOPE model."
prompt_ids = torch.tensor([sim_tokenizer.encode(prompt_text)], dtype=torch.long)

max_new_tokens = 20
generated_ids = generate_sequence(hope_model, sim_tokenizer, prompt_ids, max_new_tokens, device=device)
