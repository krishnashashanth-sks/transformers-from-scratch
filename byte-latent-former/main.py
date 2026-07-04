import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset import CustomByteDataset
from model import ByteLatentTransformer
from train import train_loop
from utils import preprocess_sequence,reconstruct_sequence
from inference import generate_sequence

# Define special tokens and their integer IDs
PAD_TOKEN = 0   # Padding token
SOS_TOKEN = 1   # Start of Sequence token
EOS_TOKEN = 2   # End of Sequence token

# Sample Data
sample_data = [
    "This is a sample sentence.",
    "Hello world!",
    "A shorter text.",
    "A very long sentence that needs to be truncated to fit the maximum sequence length defined for our Byte Latent Transformer model. This text is intentionally made long to test the truncation logic correctly. We need to ensure that the preprocessing handles various lengths effectively."
]

# Define example parameters for the ByteLatentTransformer
vocab_size = 256  # For byte-level, typically 256
d_model = 512     # Dimension of the model (embedding size)
num_heads = 8     # Number of attention heads
d_ff = 2048       # Dimension of the feed-forward network
num_encoder_layers = 2 # Number of encoder layers (reduced for quick demo)
num_decoder_layers = 2 # Number of decoder layers (reduced for quick demo)
d_latent = 128    # Dimension of the latent space
dropout_rate = 0.1
max_len = 128    # Maximum sequence length (reduced for quick demo)
eps = 1e-6        # Epsilon for LayerNorm

# Preprocess sample data
preprocessed_sequences = []
for text in sample_data:
    preprocessed_seq = preprocess_sequence(text, max_len, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)
    preprocessed_sequences.append(preprocessed_seq)

# 1. Define a batch_size
batch_size = 2 # Reduced batch size for small dataset

# 2. Create an instance of CustomByteDataset
dataset = CustomByteDataset(preprocessed_sequences, PAD_TOKEN)

# 3. Split the dataset into training and validation sets
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

if train_size == 0 and len(dataset) > 0: # Ensure at least one training sample if dataset exists
    train_size = 1
    val_size = len(dataset) - 1
if val_size == 0 and len(dataset) > 1: # Ensure at least one validation sample if dataset has more than 1
    val_size = 1
    train_size = len(dataset) - 1

if len(dataset) == 0:
    train_dataset, val_dataset = [], []
elif train_size == 0 and val_size == 0 and len(dataset) > 0:
    train_dataset, val_dataset = [dataset[0]], []
elif train_size == 0:
    train_dataset, val_dataset = [], [dataset[0]]
elif val_size == 0:
    train_dataset, val_dataset = [dataset[0]], []
else:
    # Only perform random_split if both train and val sizes are positive
    if train_size > 0 and val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    elif train_size > 0:
        train_dataset, val_dataset = dataset, [] # Use all for train if no val
    else:
        train_dataset, val_dataset = [], dataset # Use all for val if no train

# 4. Instantiate DataLoader for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if len(train_dataset) > 0 else []
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else []


# 5. Training Components

# Instantiate the ByteLatentTransformer
model = ByteLatentTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    d_latent=d_latent,
    dropout_rate=dropout_rate,
    max_len=max_len,
    eps=eps
)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# Define training parameters
epochs = 1  # For demonstration, typically much higher
kl_weight = 0.001 # Weight for KL divergence term in VAE loss
log_interval = 1 # Log every 1 batch for small dataset

print("Initiating training process...")
history = train_loop(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    pad_token=PAD_TOKEN,
    device=device,
    epochs=epochs,
    kl_weight=kl_weight,
    log_interval=log_interval
)
print("Training process completed. Loss history stored.")

# 7. Demonstrate Inference and Generation

# Reconstruction
input_sequence_tensor_recon = preprocessed_sequences[0]
original_string_recon = sample_data[0]

print(f"\nOriginal string for reconstruction: \"{original_string_recon}\"")
reconstructed_string = reconstruct_sequence(
    model=model,
    input_sequence_tensor=input_sequence_tensor_recon,
    max_len=max_len,
    sos_token=SOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    device=device
)
print(f"Reconstructed string: \"{reconstructed_string}\"")

# Generation
generated_string = generate_sequence(
    model=model,
    d_latent=d_latent,
    max_len=max_len,
    sos_token=SOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    device=device,
    latent_code=None
)
print(f"Generated string: \"{generated_string}\"")