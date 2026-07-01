from sklearn.cluster import MiniBatchKMeans
import kaggle
import os
from model import *
from dataset import HuBERTDataset
from losses import hubert_loss
import torch
from torch.utils.data import DataLoader
from utils import *
import matplotlib.pyplot as plt
import tqdm

dataset_ref = 'paultimothymooney/medical-speech-transcription-and-intent'
output_dir = 'medical-speech-data'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading dataset: {dataset_ref} to {output_dir}")
kaggle.api.dataset_download_files(dataset_ref, path=output_dir, unzip=True)
print("Download and extraction complete.")

# List the contents of the downloaded directory to verify
print(f"Contents of '{output_dir}':")

base_data_path = 'medical-speech-data/Medical Speech, Transcription, and Intent'
recordings_path = os.path.join(base_data_path, 'recordings')
csv_path = 'medical-speech-data/Medical Speech, Transcription, and Intent/overview-of-recordings.csv'


if os.path.exists(csv_path) and os.path.exists(base_data_path):
  print("\n--- Initializing HuBERTDataset ---")
  # For now, we'll initialize without discrete targets.
  # Targets will be assigned after initial clustering of the entire dataset.
  hubert_dataset = HuBERTDataset(
      data_root=base_data_path,
      metadata_csv_path=csv_path,
      sample_rate=16000,
      n_mels=80,
      n_fft=400,
      hop_length=160
  )

  print("\n--- Initializing DataLoader ---")
  batch_size = 8 # Example batch size
  hubert_dataloader = DataLoader(
      hubert_dataset,
      batch_size=batch_size,
      shuffle=True, # Shuffle for training
      collate_fn=collate_fn_hubert,
      num_workers=2 # Use multiple workers for faster data loading
  )

  print(f"DataLoader created with batch size {batch_size} and {hubert_dataloader.num_workers} workers.")

  # Fetch a sample batch to inspect its structure
  print("\n--- Fetching a sample batch ---")
  for batch_idx, (mel_features_batch, indices_batch, lengths_batch) in enumerate(hubert_dataloader):
    print(f"Batch {batch_idx+1}:")
    print(f"  Mel Features Batch Shape: {mel_features_batch.shape}") # (batch_size, n_mels, max_len)
    print(f"  Indices Batch Shape: {indices_batch.shape}")           # (batch_size,) - original indices in dataset
    print(f"  Lengths Batch Shape: {lengths_batch.shape}")           # (batch_size,) - original lengths of features
    print(f"  First 3 lengths: {lengths_batch[:3].tolist()}")
    print(f"  Max length in batch: {mel_features_batch.shape[2]}")

    # Stop after the first batch for demonstration
    break

  print("\nNote: At this stage, the DataLoader returns Mel features and original dataset indices. ")
  print("The next step will be to extract features for the *entire* dataset and perform initial clustering to get the discrete targets, which will then be assigned back to the dataset.")

else:
  print("Dataset paths not found. Please ensure Kaggle data download and extraction were successful.")

all_features = []
all_lengths = []

with torch.no_grad():
        for batch_idx, (mel_features_batch, indices_batch, lengths_batch) in enumerate(hubert_dataloader):
            # Move features to CPU for clustering (sklearn works with numpy/CPU tensors)
            all_features.append(mel_features_batch.cpu().permute(0, 2, 1).reshape(-1, mel_features_batch.shape[1]))
            all_lengths.extend(lengths_batch.tolist())

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(hubert_dataloader)} batches for feature extraction.")


flattened_features_list = []
current_feature_idx = 0
for i, length in enumerate(all_lengths):
    # Reshape to get (time_steps, n_mels) for the current audio file
    single_audio_features = all_features[i // hubert_dataloader.batch_size][(i % hubert_dataloader.batch_size) * hubert_dataset.n_mels : ((i % hubert_dataloader.batch_size) + 1) * hubert_dataset.n_mels].view(-1, hubert_dataset.n_mels)
    flattened_features_list.append(single_audio_features[:length])

    # This part needs careful adjustment due to how all_features was constructed
    # Let's rebuild all_features correctly as a flat list of actual feature frames
correct_flattened_features = []
start_idx = 0

temp_dataloader = DataLoader(
        hubert_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_hubert,
        num_workers=2
    )

temp_dataloader_eval = DataLoader(
            hubert_dataset, # Use the same dataset, which has assigned targets
            batch_size=1, # Process one by one to avoid padding issues during feature concatenation
            shuffle=False,
            collate_fn=collate_fn_hubert,
            num_workers=2
        )

all_unpadded_features = []
all_original_lengths = []
print("Collecting unpadded features for clustering...")
for i, (mel_features_batch, indices_batch, lengths_batch) in enumerate(temp_dataloader):
    # mel_features_batch is (1, n_mels, time_steps)
    # We take only the actual length, and reshape to (time_steps, n_mels)
    actual_length = lengths_batch.item()
    unpadded_feature = mel_features_batch.squeeze(0)[:, :actual_length].permute(1, 0)
    all_unpadded_features.append(unpadded_feature)
    all_original_lengths.append(actual_length)
    if (i + 1) % 1000 == 0:
        print(f"  Collected features for {i + 1}/{len(hubert_dataset)} audio files.")

all_unpadded_features_tensor = torch.cat(all_unpadded_features, dim=0)
print(f"Total frames for clustering: {all_unpadded_features_tensor.shape[0]}")
num_clusters=100
# Perform K-Means clustering
print(f"Clustering {all_unpadded_features_tensor.shape[0]} frames with {num_clusters} clusters...")
kmeans_full_dataset = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, n_init=10, batch_size=256, verbose=False)
kmeans_full_dataset.fit(all_unpadded_features_tensor.cpu().numpy())
full_discrete_targets = kmeans_full_dataset.labels_

print(f"Generated {len(full_discrete_targets)} discrete targets.")
print(f"Number of unique clusters found: {len(np.unique(full_discrete_targets))}")

# Assign discrete targets back to the dataset
hubert_dataset.assign_discrete_targets(full_discrete_targets, all_original_lengths)

# Now, redefine the dataloader to return features and *assigned* targets
# Shuffle for training
hubert_dataloader_with_targets = DataLoader(
    hubert_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_hubert,
    num_workers=2
)

num_iterations = 2 # Number of full HuBERT iterations (training + re-clustering)
epochs_per_iteration = 3 # Number of training epochs within each iteration

input_dim = 80 # n_mels
model_dim = 256 # Dimension of the hidden states (e.g., 256 or 768 for larger models)
num_heads = 4   # Number of attention heads
num_layers = 2

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer_model = TransformerEncoder(input_dim, model_dim, num_heads, num_layers)
prediction_head = PredictionHead(model_dim=model_dim, num_clusters=num_clusters)

# Ensure models are on the correct device
transformer_model.to(device)
prediction_head.to(device)


# Phase 2: Extract Contextualized Features for Re-clustering
print("Extracting features for re-clustering...")
transformer_model.eval() # Set model to evaluation mode

all_extracted_features = []
all_original_lengths_for_recluster = [] # Need to re-collect lengths as dataloader order changes

# Use a temporary dataloader with shuffle=False to ensure consistent order for feature extraction
temp_dataloader_eval = DataLoader(
    hubert_dataset, # Use the same dataset, which has assigned targets
    batch_size=1, # Process one by one to avoid padding issues during feature concatenation
    shuffle=False,
    collate_fn=collate_fn_hubert,
    num_workers=2
)

with torch.no_grad():
    for batch_idx, (mel_features_batch_eval, _, lengths_batch_eval) in enumerate(tqdm.tqdm(temp_dataloader_eval, desc="Feature Extraction")):
        # Move features to device
        mel_features_batch_eval = mel_features_batch_eval.to(device).float()

        # Pass *original* (unmasked) features through the transformer to get new representations
        # Output shape: (batch_size, time_steps, model_dim)
        extracted_features_batch = transformer_model(mel_features_batch_eval)

        # Take only the actual length of features
        actual_length = lengths_batch_eval.item()
        unpadded_features = extracted_features_batch.squeeze(0)[:actual_length, :]
        all_extracted_features.append(unpadded_features)
        all_original_lengths_for_recluster.append(actual_length)

        if (batch_idx + 1) % 1000 == 0:
            print(f"  Extracted features for {batch_idx + 1}/{len(hubert_dataset)} audio files.")

# Set the model to evaluation mode
transformer_model.eval()

# Get a single audio sample from the dataset for inference
# We'll take the first item, but you could load any new audio file.
# The __getitem__ method returns (mel_features, targets) at this stage.
sample_mel_features, sample_targets = hubert_dataset[0]

# Add a batch dimension and move to device
# Expected shape for transformer_model: (batch_size, n_mels, time_steps)
input_features_for_inference = sample_mel_features.unsqueeze(0).to(device).float()

print(f"Input features shape for inference: {input_features_for_inference.shape}")

with torch.no_grad(): # Disable gradient calculations for inference
    # Pass the input features through the transformer model
    # The model expects (batch_size, time_steps, n_mels) after input_projection,
    # so the permute happens inside the model's forward method.
    inferred_features = transformer_model(input_features_for_inference)



# Convert tensors to CPU and numpy for plotting
input_np = input_features_for_inference.squeeze(0).cpu().numpy()
inferred_np = inferred_features.squeeze(0).cpu().numpy()

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_np, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Original Mel Features (Input)')
plt.xlabel('Time Frames')
plt.ylabel('Mel Bins')

plt.subplot(1, 2, 2)
# The inferred features are (time_steps, model_dim), so we don't need to permute for imshow
plt.imshow(inferred_np.T, origin='lower', aspect='auto', cmap='magma') # Transpose for better visualization
plt.colorbar()
plt.title('Inferred Contextualized Features (Output)')
plt.xlabel('Time Frames')
plt.ylabel('Feature Dimension')

plt.tight_layout()
plt.show()
