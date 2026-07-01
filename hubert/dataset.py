from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from utils import extract_mel_features_from_waveform
import os

class HuBERTDataset(Dataset):
  def __init__(self, data_root, metadata_csv_path, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
    self.data_root = data_root
    self.metadata_csv_path = metadata_csv_path
    self.sample_rate = sample_rate
    self.n_mels = n_mels
    self.n_fft = n_fft
    self.hop_length = hop_length

    # Load metadata
    self.metadata_df = pd.read_csv(metadata_csv_path)
    # Filter out rows where file_name might be missing or invalid
    self.metadata_df = self.metadata_df.dropna(subset=['file_name'])

    self.audio_paths = []
    for index, row in self.metadata_df.iterrows():
      # Assuming audio files are directly in data_root (e.g., recordings/train, recordings/test, recordings/validate)
      # We need to correctly identify the subdirectory (train/test/validate) for each file.
      # The 'file_name' column directly contains the WAV file name.
      # The recordings are split into train/test/validate within the data_root structure.

      # This is a simplification; a more robust solution would map file_name to its actual path.
      # For this dataset, the recordings folder itself contains train/test/validate.
      # Let's assume for now we only care about the 'train' split for pre-training.
      # A more general approach would require iterating through subdirs or having this info in CSV.

      # Let's find the actual path by searching
      file_name = row['file_name']
      # Search within train, test, validate directories
      found_path = None
      for split_dir in ['train', 'test', 'validate']:
        potential_path = os.path.join(data_root, split_dir, file_name)
        if os.path.exists(potential_path):
          found_path = potential_path
          break

      if found_path:
        self.audio_paths.append(found_path)
      else:
        print(f"Warning: Audio file not found for {file_name}. Skipping.")

    # Placeholder for discrete targets. These will be assigned after initial clustering.
    self.discrete_targets = None # This will be a list of tensors/arrays, one per audio file, or None
    self.audio_lengths = [] # To store original feature lengths for target assignment

    if not self.audio_paths:
      raise ValueError("No valid audio files found in the specified data_root and metadata.")

    print(f"Initialized HuBERTDataset with {len(self.audio_paths)} audio files.")

  def __len__(self):
    return len(self.audio_paths)

  def __getitem__(self, idx):
    audio_path = self.audio_paths[idx]
    waveform, sr = torchaudio.load(audio_path)

    # Extract Mel features (output will be n_mels, time_steps)
    mel_features = extract_mel_features_from_waveform(
        waveform, sr,
        target_sample_rate=self.sample_rate, n_mels=self.n_mels,
        n_fft=self.n_fft, hop_length=self.hop_length
    )

    # Optionally store the length for target mapping later
    # self.audio_lengths.append(mel_features.shape[1]) # This is tricky in __getitem__ with multiprocessing.
    # It's better to get lengths in a separate pass or within collate_fn if possible for targets.

    # If discrete targets are assigned, return them. Otherwise, return a placeholder.
    if self.discrete_targets is not None:
      # discrete_targets is a list of tensors/arrays, one per audio
      targets = self.discrete_targets[idx]
      return mel_features, targets
    else:
      # Return features and index for initial clustering phase (where targets are not yet available)
      return mel_features, idx # return index to map targets back after clustering

  def assign_discrete_targets(self, all_discrete_targets, all_audio_lengths):
    """
    Assigns pre-computed discrete targets to the dataset.
    Args:
        all_discrete_targets (np.ndarray): A flat array of all discrete targets.
        all_audio_lengths (list): A list of feature lengths for each audio file in order.
    """
    if len(all_audio_lengths) != len(self.audio_paths):
      raise ValueError("Number of audio lengths does not match number of audio files.")

    self.discrete_targets = []
    current_idx = 0
    for length in all_audio_lengths:
      self.discrete_targets.append(all_discrete_targets[current_idx : current_idx + length])
      current_idx += length
    print(f"Assigned discrete targets to {len(self.discrete_targets)} audio files.")