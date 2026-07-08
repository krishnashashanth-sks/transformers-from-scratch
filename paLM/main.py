import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from model import ConceptualPaLModel
from dataset import ConceptualMultimodalDataset
from transformers import AutoTokenizer
import torch
from utils import real_tokenize_multilingual
from train import train_model
from inference import infer_model

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

# Define hyperparameters (consistent with previous cells)
text_vocab_size = tokenizer.vocab_size # Using the tokenizer from previous cell
text_embed_dim = 256
image_feature_dim = 768
common_embed_dim = 512
max_text_seq_len = 128
num_image_tokens = 2
max_seq_len = max_text_seq_len + num_image_tokens # Total combined sequence length
num_transformer_layers = 4
num_heads = 8
ff_dim = 2048
output_dim = 1000 # Example output dimension for a classification/next token task

# Instantiate the model (assuming `conceptual_palm` is available from previous cell)
# If not, uncomment and run the following line:
conceptual_palm = ConceptualPaLModel(
    text_vocab_size=text_vocab_size,
    text_embed_dim=text_embed_dim,
    image_feature_dim=image_feature_dim,
    common_embed_dim=common_embed_dim,
    max_seq_len=max_seq_len,
    num_transformer_layers=num_transformer_layers,
    num_heads=num_heads,
    ff_dim=ff_dim,
    output_dim=output_dim
)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW(conceptual_palm.parameters(), lr=0.001)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a larger set of dummy data for the DataLoader
dummy_training_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "El veloz zorro marrón salta sobre el perro perezoso.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "Eine schnelle braune Fuchs springt über den faulen Hund.",
    "The cat sat on the mat.",
    "El gato se sentó en la alfombra.",
    "Le chat s'est assis sur le tapis.",
    "Die Katze saß auf der Matte.",
    "A bird flew high in the sky.",
    "Un pájaro voló alto en el cielo."
] * 5 # Make it 50 samples for a slightly larger dataset

batch_size = 4
num_epochs = 3

# Instantiate the dataset and DataLoader
training_dataset = ConceptualMultimodalDataset(
    text_samples=dummy_training_texts,
    image_feature_dim=image_feature_dim,
    num_image_tokens=num_image_tokens,
    max_text_seq_len=max_text_seq_len,
    output_dim=output_dim,
    tokenizer_func=real_tokenize_multilingual,
    tokenizer=tokenizer
)

training_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset created with {len(training_dataset)} samples.")
print(f"DataLoader created with batch size {batch_size}. There are {len(training_data_loader)} batches.")

# --- Run Training ---
train_model(conceptual_palm, training_data_loader, loss_function, optimizer, num_epochs, device)

# --- Run Inference ---
inference_predictions = infer_model(conceptual_palm, training_data_loader, device) # Using training loader for inference for simplicity

print(f"\nShape of inference predictions (first 5 samples):\n{inference_predictions[:5].shape}")
print(f"Example prediction for the first sequence (first batch, first sample):\n{inference_predictions[0][:max_text_seq_len + num_image_tokens]}")
