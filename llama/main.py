from inference import generate_llama_text
import time
from evaluate import evaluate_step_llama
from train import train_step_llama
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from dataset import LlamaCausalLMDataset
from utils import tokenize
from model import *

sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The sun rises in the east every morning.",
    "Water is essential for all living things.",
    "Artificial intelligence is changing the world.",
    "Machine learning algorithms can analyze vast amounts of data.",
    "Natural language processing helps computers understand human language.",
    "Deep learning models require significant computational resources.",
    "Python is a popular programming language for data science.",
    "TensorFlow and PyTorch are leading deep learning frameworks.",
    "The Earth orbits the sun in an elliptical path.",
    "Birds migrate south for the winter to find warmer climates.",
    "Books offer a window into different worlds and ideas.",
    "Coding helps develop problem-solving skills.",
    "Music can evoke a wide range of emotions and memories."
]

special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
vocab = {token: i for i, token in enumerate(special_tokens)}

pad_id=vocab['[PAD]']

# Extract all unique words from sample_texts and convert to lowercase
all_words = []
for text in sample_texts:
    all_words.extend(text.lower().replace('.', '').split())

unique_words = sorted(list(set(all_words)))

# Add unique words to the vocabulary, starting after special tokens
for word in unique_words:
    if word not in vocab:
        vocab[word] = len(vocab)

# Create an inverse mapping from ID to token
idx_to_token = {idx: token for token, idx in vocab.items()}

learning_rate_llama = 1e-4
num_train_epochs_llama = 10 # Example number of epochs
weight_decay_llama = 0.01

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

vocab_size_llama = 1000
dim_llama = 64          # Embedding dimension
num_layers_llama = 2    # Number of decoder blocks
num_heads_llama = 2     # Number of attention heads
hidden_size_llama = 128 # Hidden dimension for SwiGLU FFN (typically 4*dim or similar)
max_seq_len_llama = 50
dropout_rate_llama = 0.1

llama_model_instance = Llama(
    vocab_size=vocab_size_llama,
    dim=dim_llama,
    num_layers=num_layers_llama,
    num_heads=num_heads_llama,
    hidden_size=hidden_size_llama,
    max_seq_len=max_seq_len_llama,
    dropout_rate=dropout_rate_llama
)

# Instantiate the LlamaForCausalLM model
llama_model_for_training = LlamaForCausalLM(
    llama_model_instance, # Use the previously created Llama model backbone
    vocab_size_llama
)

# Move model to device if GPU is available (assuming 'device' is already defined)
llama_model_for_training.to(device)

clm_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# Instantiate the AdamW optimizer
llama_optimizer = optim.AdamW(
    llama_model_for_training.parameters(),
    lr=learning_rate_llama,
    weight_decay=weight_decay_llama
)

llama_dataset = LlamaCausalLMDataset(
    sample_texts=sample_texts,
    max_seq_len=max_seq_len_llama,
    pad_id=pad_id
)

print(f"Length of LlamaCausalLMDataset: {len(llama_dataset)} samples")

batch_size = 4 # Define a batch size for the DataLoader
llama_dataloader = DataLoader(
    llama_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0 # Set to 0 for simplicity in Colab, consider >0 for performance in real apps
)

num_eval_samples_llama = len(sample_texts) # Use all sample texts for evaluation in this small example
# Instantiate a LlamaCausalLMDataset for evaluation
llama_eval_dataset = LlamaCausalLMDataset(
    sample_texts=sample_texts,
    max_seq_len=max_seq_len_llama,
    pad_id=pad_id
)

# Instantiate a DataLoader for the evaluation dataset
llama_eval_dataloader = DataLoader(
    llama_eval_dataset,
    batch_size=batch_size, # Re-using batch_size from training DataLoader
    shuffle=False, # Do not shuffle for evaluation
    num_workers=0
)

# 1. Initialize an empty list to store the training history
llama_train_history = []

print("Starting Llama pre-training...")

# 2. Start a loop that iterates num_train_epochs_llama times
for epoch in range(num_train_epochs_llama):
    start_time = time.time()

    # 3. Inside the epoch loop, initialize variables to accumulate losses for the current epoch
    total_epoch_loss_llama = 0
    num_batches_llama = 0

    # 4. Start an inner loop that iterates through each batch in the llama_dataloader
    for i, batch in enumerate(llama_dataloader):
        # 5. Call the train_step_llama function
        batch_loss = train_step_llama(
            llama_model_for_training, batch, clm_criterion, llama_optimizer, device
        )

        # 6. Accumulate the batch loss and increment batch counter
        total_epoch_loss_llama += batch_loss
        num_batches_llama += 1

    # 7. After the inner batch loop finishes, calculate the average epoch loss
    avg_epoch_loss_llama = total_epoch_loss_llama / num_batches_llama

    # 8. Append this average epoch loss to the history list
    llama_train_history.append(avg_epoch_loss_llama)

    end_time = time.time()
    epoch_duration = end_time - start_time

    # 9. Print the epoch number and average loss periodically
    print(f"Epoch {epoch+1}/{num_train_epochs_llama} | Duration: {epoch_duration:.2f}s | Avg CLM Loss: {avg_epoch_loss_llama:.4f}")

print("Llama pre-training finished.")


print("Starting Llama evaluation...")

# 1. Initialize variables to accumulate losses
total_eval_clm_loss = 0
num_eval_batches = 0

# 2. Set the model to evaluation mode (already done in evaluate_step_llama, but good practice)
llama_model_for_training.eval()

# 3. Iterate through each batch in the llama_eval_dataloader
for i, batch in enumerate(llama_eval_dataloader):
    # 4. Call the evaluate_step_llama function
    batch_clm_loss = evaluate_step_llama(
        llama_model_for_training, batch, clm_criterion, device
    )

    # 5. Accumulate the returned losses
    total_eval_clm_loss += batch_clm_loss
    num_eval_batches += 1

# 6. Calculate average evaluation CLM loss
avg_eval_clm_loss = total_eval_clm_loss / num_eval_batches

# 7. Print the calculated average evaluation CLM loss
print("Llama evaluation finished.")
print(f"\nEvaluation Results: Avg CLM Loss: {avg_eval_clm_loss:.4f}")

print("\n--- Testing Llama Text Generation ---")

# Sample prompt for generation
# The vocabulary is limited, so prompts should use words from the sample_texts
llama_prompt = "the quick brown"
max_generation_length = 20 # Max tokens to generate after the prompt

# Ensure the model is on the correct device
llama_model_for_training.to(device)

generated_text = generate_llama_text(
    model=llama_model_for_training,
    tokenizer=tokenize, # Our custom tokenizer
    idx_to_token=idx_to_token,
    vocab=vocab,
    prompt=llama_prompt,
    max_gen_len=max_generation_length,
    device=device
)

print(f"Prompt: '{llama_prompt}'")
print(f"Generated Text: '{generated_text}'")

print("\nNote: The quality of generated text will be very low due to the small vocabulary, small dataset, and limited training epochs. This is purely for demonstrating the generation process.")