import torch.optim as optim
from dataset import TextClassificationDataset
from config import LongformerConfig
from model import LongformerModel
import torch
import random
from tokenizer import SimpleWordTokenizer
import torch.nn as nn
from train import train
from inference import predict_text

# 1. Generate a dummy dataset
dummy_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "The only way to do great work is to love what you do.",
    "Innovation distinguishes between a leader and a follower.",
    "The best way to predict the future is to create it.",
    "Stay hungry, stay foolish.",
    "Success is not final, failure is not fatal: it is the courage to continue that counts."
]

dummy_labels = [random.randint(0, 2) for _ in range(len(dummy_texts))] # 3 classes: 0, 1, 2

# 2. Initialize and build the vocabulary for the SimpleWordTokenizer
# Re-initialize the tokenizer to ensure a clean vocabulary build with the dummy_texts
tokenizer = SimpleWordTokenizer()
tokenizer.build_vocabulary(dummy_texts)

print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
longformer_config = LongformerConfig(
    vocab_size=len(tokenizer.vocab), # Use vocab size from our custom tokenizer
    hidden_size=256,                 # Smaller hidden size for demonstration
    num_hidden_layers=2,             # Fewer layers for quicker instantiation
    num_attention_heads=4,           # Fewer attention heads
    intermediate_size=512,           # Smaller intermediate size
    attention_window=64,             # Smaller attention window
    max_position_embeddings=1024 # Use the max_seq_len from tokenizer test
)
print("LongformerConfig instantiated with custom parameters.")

dataset = TextClassificationDataset(dummy_texts, dummy_labels, tokenizer, longformer_config.max_position_embeddings)

from torch.utils.data import DataLoader
batch_size=2
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)

# 2. Create an instance of the LongformerModel
model = LongformerModel(longformer_config)
print("LongformerModel instantiated.")

# 3. Initialize the AdamW optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
print("AdamW optimizer initialized.")

# 4. (Optional) Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to: {device}")

# 5.Instantiate nn.CrossEntropyLoss for multi-class classification
loss_fn = nn.CrossEntropyLoss()

#6.Training
num_epochs=10

train_losses,eval_losses,eval_accuracies=train(num_epochs,model,dataloader,optimizer,loss_fn,device)

print(f"Final Training Losses: {train_losses}")
print(f"Final Evaluation Losses: {eval_losses}")
print(f"Final Evaluation Accuracies: {eval_accuracies}")

print("\n--- Demonstrating Text Classification on New Inputs ---")

# Example texts for prediction
new_texts = [
    "This is a completely new sentence for classification.",
    "The future is bright and full of possibilities.",
    "A quick fox jumps over the lazy dog."
]

for text in new_texts:
    predicted_id, probabilities = predict_text(text, model, tokenizer, longformer_config.max_position_embeddings, device)
    print(f"Input Text: '{text}'")
    print(f"Predicted Class ID: {predicted_id}")
    print(f"Probabilities per class: {probabilities}")