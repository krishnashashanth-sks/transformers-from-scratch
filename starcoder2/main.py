import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
from tokenizer import BasicBPETokenizer
from config import StarCoderV2Config
from model import StarCoderV2ForCausalLM
from train import train_model

#1. Initialize the bpe tokenzier
print("\n--- Training BPE Tokenizer ---")
corpus = "the quick brown fox jumps over the lazy dog dog dog quick quick brown fox brown. This is a sample text for training the tokenizer. We need enough text to learn merges efficiently. Long sentences are helpful to demonstrate tokenization."
bpe_tokenizer = BasicBPETokenizer(special_tokens=['</w>', '<unk>'])
bpe_tokenizer.train(corpus, num_merges=100)
print(f"BPE Tokenizer vocabulary size: {len(bpe_tokenizer.token_to_id)}")

# --- 2. Initialize Model Configuration ---
print("\n--- Initializing StarCoderV2 model configuration ---")
config = StarCoderV2Config(
    vocab_size=len(bpe_tokenizer.token_to_id), # Use the vocabulary size from the BPE tokenizer
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2, # GQA with 2 K/V heads
    intermediate_size=1024,
    max_position_embeddings=512,
    dropout_rate=0.05,
    norm_epsilon=1e-5
)

# --- 3. Instantiate Model ---
model = StarCoderV2ForCausalLM(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- 4. Prepare Dataset for Training ---
print("\n--- Preparing Dataset for Training ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Load a small dataset for demonstration
print("Loading wikitext-2-raw-v1 dataset (small sample for training demo)... ")
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]") # Load 1000 examples

def bpe_tokenize_function(examples):
    tokenized_texts = []
    for text in examples["text"]:
        if text is None or not text.strip():
            tokenized_texts.append([])
            continue
        tokenized_texts.append(bpe_tokenizer.encode(text))
    return {"input_ids_list": tokenized_texts}

tokenized_datasets = raw_datasets.map(
    bpe_tokenize_function,
    batched=True,
    remove_columns=raw_datasets.column_names,
    desc="Tokenizing dataset with BPE tokenizer",
)

block_size = config.max_position_embeddings
def group_texts(examples):
    concatenated_input_ids = sum(examples["input_ids_list"], [])
    total_length = (len(concatenated_input_ids) // block_size) * block_size
    chunked_input_ids = []
    chunked_labels = []
    for i in range(0, total_length, block_size):
        chunked_input_ids.append(concatenated_input_ids[i : i + block_size])
        chunked_labels.append(concatenated_input_ids[i : i + block_size])
    return {"input_ids": chunked_input_ids, "labels": chunked_labels}

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    remove_columns=tokenized_datasets.column_names,
    desc=f"Grouping texts in chunks of {block_size}",
)
lm_datasets = lm_datasets.filter(lambda x: len(x["input_ids"]) == block_size)
lm_datasets.set_format(type="torch", columns=["input_ids", "labels"])

train_batch_size = 2
train_dataloader = DataLoader(lm_datasets, shuffle=True, batch_size=train_batch_size)

# --- 5. Training Loop ---
print("\n--- Starting Training ---")
learning_rate = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
num_epochs = 1

train_model(model,num_epochs,train_dataloader,optimizer,device)

# --- 6. Inference Loop (Text Generation) ---
print("\n--- Starting Inference (Text Generation) ---")
model.eval()

start_text = "the quick brown"
start_token_ids = bpe_tokenizer.encode(start_text)
input_prompt_ids = torch.tensor([start_token_ids], device=device)

max_new_tokens = 50
generated_ids = input_prompt_ids

print(f"Starting generation with input: '{start_text}'")

with torch.no_grad():
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated_ids, device=device)
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

decoded_generated_text = bpe_tokenizer.decode(generated_ids.squeeze().tolist())
print(f"Decoded Generated Text: '{decoded_generated_text}'")
print("Inference complete!\n")