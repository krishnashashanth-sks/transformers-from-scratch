import torch
import torchaudio
import numpy as np

def collate_fn_hubert(batch):
  """
  Custom collate function for HuBERTDataset. Pads features and targets to the longest
  sequence in the batch.

  Args:
      batch (list): A list of samples, where each sample is (mel_features, targets)
                    or (mel_features, idx) during initial target generation.

  Returns:
      tuple: Padded mel_features, padded targets (if available), and original lengths.
  """
  # Assuming batch elements are (mel_features, targets) or (mel_features, idx)
  mel_features_list = [item[0] for item in batch]
  metadata_list = [item[1] for item in batch] # This could be targets or original indices

  lengths = torch.tensor([f.shape[1] for f in mel_features_list], dtype=torch.long)
  max_len = max(lengths)

  padded_mel_features = []
  for mel_feature in mel_features_list:
    # mel_feature shape: (n_mels, time_steps)
    # Pad along the time_steps dimension
    padding = (0, max_len - mel_feature.shape[1]) # (pad_left, pad_right) for last dim
    padded_feature = torch.nn.functional.pad(mel_feature, padding, "constant", 0)
    padded_mel_features.append(padded_feature)

  padded_mel_features = torch.stack(padded_mel_features) # Result: (batch_size, n_mels, max_len)

  if isinstance(metadata_list[0], torch.Tensor) or isinstance(metadata_list[0], np.ndarray):
    # We have targets, need to pad them too
    padded_targets = []
    for targets in metadata_list:
      if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).long()
      padding = (0, max_len - targets.shape[0])
      padded_target = torch.nn.functional.pad(targets, padding, "constant", -1) # Use -1 for padding targets
      padded_targets.append(padded_target)
    padded_targets = torch.stack(padded_targets)
    return padded_mel_features, padded_targets, lengths
  else:
    # We have indices (for initial clustering), just return them
    indices = torch.tensor(metadata_list, dtype=torch.long)
    return padded_mel_features, indices, lengths


# Original extract_mel_features function (slightly modified to accept waveform directly)
def extract_mel_features_from_waveform(waveform, sr, target_sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
  """
  Extracts Mel-filter bank energies (FBanks) from an audio waveform.

  Args:
      waveform (torch.Tensor): Raw audio waveform (channel, samples).
      sr (int): Original sample rate of the waveform.
      target_sample_rate (int): Desired sample rate for audio processing.
      n_mels (int): Number of Mel bands to generate.
      n_fft (int): Size of FFT, creates window_size of n_fft. (e.g., for 16kHz audio, 400 samples = 25ms)
      hop_length (int): Number of samples between successive frames. (e.g., for 16kHz audio, 160 samples = 10ms)

  Returns:
      torch.Tensor: Log Mel-filter bank energies tensor (n_mels, time_steps).
  """
  # Resample if sample rate does not match
  if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    waveform = resampler(waveform)

  # Ensure mono audio (if stereo, take the first channel)
  if waveform.shape[0] > 1:
    waveform = waveform[0, :].unsqueeze(0)

  # Initialize MelSpectrogram transform
  mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
      sample_rate=target_sample_rate,
      n_fft=n_fft,
      hop_length=hop_length,
      n_mels=n_mels
  )

  # Apply transform and convert to log scale for better distribution
  mel_features = mel_spectrogram_transform(waveform)
  log_mel_features = torchaudio.transforms.AmplitudeToDB()(mel_features)

  # Squeeze the batch dimension if it's 1, result in (n_mels, time_steps)
  return log_mel_features.squeeze(0)

import torch.nn.functional as F
def apply_span_masking(features,mask_prob=0.15,mask_length=10,special_mask_token=None):
  """
  Applies span masking to the input features.

  Args:
      features (torch.Tensor): Input features, shape (batch_size, n_mels, time_steps).
      mask_prob (float): Probability of masking a given frame.
      mask_length (int): Average length of the masked spans.
      special_mask_token (torch.Tensor, optional): A special tensor to replace masked spans.
                                                If None, zeros will be used.

  Returns:
      torch.Tensor: Masked features.
      torch.BoolTensor: A boolean tensor indicating masked positions (True for masked).
  """
  batch_size,n_mels,time_steps=features.shape
  mask=torch.zeros((batch_size,time_steps),dtype=torch.bool,device=features.device)
  num_masked_frames=int(time_steps*mask_prob)
  masked_indices=torch.randperm(time_steps,device=features.device)[:num_masked_frames]
  for idx in masked_indices:
    # Convert 0-d tensor 'idx' to a Python scalar using .item()
    start=max(0, idx.item() - mask_length//2)
    end=min(time_steps, idx.item() + mask_length//2 + 1)
    mask[:,start:end]=True
  masked_features=features.clone()
  if special_mask_token is not None:
    # special_mask_token should have shape (1, n_mels, 1) to broadcast correctly
    masked_features.masked_fill_(mask.unsqueeze(1).expand_as(masked_features), 0.) # replace with 0s first
    masked_features.masked_fill_(mask.unsqueeze(1).expand_as(masked_features), special_mask_token) # then with mask token if provided
  else:
    masked_features.masked_fill_(mask.unsqueeze(1).expand_as(masked_features),0.)
  return masked_features,mask