import torch
import torch.nn as nn
import torch.optim as optim
from dataset import WikitextDataset
from model import ElectraModel
from inference import electra_inference
from train import train_model
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast

# Configuration (adjust as needed)
VOCAB_SIZE = 30522 # Example for WordPiece tokenizer, ensure this matches your tokenizer
MAX_SEQ_LEN = 128
HIDDEN_SIZE = 256 # Smaller for generator, larger for discriminator
NUM_ATTENTION_HEADS = 4
NUM_HIDDEN_LAYERS = 4
INTERMEDIATE_SIZE = HIDDEN_SIZE * 4 # Typically 4x hidden size

# ELECTRA specific parameters
MASK_PROB = 0.15 # Probability of masking a token
REPLACE_PROB = 0.85 # Probability that a masked token is replaced by generator output (vs. keeping original)
# Other common masking strategies: 0.1 for random token, 0.1 for original token

# Special token IDs (adjust based on your tokenizer)
MASK_TOKEN_ID = 103 # [MASK]
PAD_TOKEN_ID = 0    # [PAD]

# Instantiate models with updated configurations
gen_config = {
    'vocab_size': VOCAB_SIZE,
    'hidden_size': HIDDEN_SIZE,
    'num_attention_heads': NUM_ATTENTION_HEADS,
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'intermediate_size': INTERMEDIATE_SIZE
}

disc_config = {
    'vocab_size': VOCAB_SIZE,
    'hidden_size': HIDDEN_SIZE * 2, # Discriminator typically larger
    'num_attention_heads': NUM_ATTENTION_HEADS * 2,
    'num_hidden_layers': NUM_HIDDEN_LAYERS * 2,
    'intermediate_size': INTERMEDIATE_SIZE * 2
}

electra_model = ElectraModel(
    gen_config,
    disc_config,
    mask_prob=MASK_PROB,
    replace_prob=REPLACE_PROB,
    mask_token_id=MASK_TOKEN_ID,
    pad_token_id=PAD_TOKEN_ID
)

# Optimizers
gen_optimizer = optim.AdamW(electra_model.generator.parameters(), lr=1e-4)
disc_optimizer = optim.AdamW(electra_model.discriminator.parameters(), lr=1e-4)

# Loss functions
gen_loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Standard MLM loss for generator
disc_loss_fct = nn.BCEWithLogitsLoss() # Binary cross-entropy for discriminator

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load a tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Update global configuration based on the tokenizer
VOCAB_SIZE = tokenizer.vocab_size
MASK_TOKEN_ID = tokenizer.mask_token_id
PAD_TOKEN_ID = tokenizer.pad_token_id

print(f"Updated VOCAB_SIZE: {VOCAB_SIZE}")
print(f"Updated MASK_TOKEN_ID: {MASK_TOKEN_ID}")
print(f"Updated PAD_TOKEN_ID: {PAD_TOKEN_ID}")

# Tokenize the dataset
def tokenize_function(examples):
    # Ensure all examples are string, replace None with empty string
    examples['text'] = [str(x) if x is not None else "" for x in examples['text']]
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)

# Apply tokenization to the dataset. We'll process the 'train' split for now.
# Using batch=True for faster tokenization.
tokenized_datasets = dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"])

# Create an instance of our custom dataset
train_dataset = WikitextDataset(tokenized_datasets, MAX_SEQ_LEN, PAD_TOKEN_ID)

# Create a DataLoader
batch_size = 8 # Use the same batch_size as defined before in cell 45ac3d7c
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 3 # Example number of epochs
train_model(num_epochs,train_data_loader,electra_model,gen_loss_fct,disc_loss_fct,gen_optimizer,disc_optimizer,VOCAB_SIZE)

# Example usage of the inference function
sample_text = "The quick brown fox jumps over the lazy dog."

gen_inf_results, disc_inf_results = electra_inference(
    electra_model,
    tokenizer,
    sample_text,
    MAX_SEQ_LEN,
    MASK_TOKEN_ID,
    VOCAB_SIZE,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

print("\nGenerated Text (from Generator inference, masked positions only):", gen_inf_results["predicted_tokens_str"])