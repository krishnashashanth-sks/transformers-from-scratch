import torchaudio
from utils import extract_mel_features_from_waveform
import torch
import matplotlib.pyplot as plt

def perform_inference_and_visualize(audio_file_path,transformer_model,input_dim,device):
    if audio_file_path is None:
        return "Please upload an audio file.", None, None
    
    # 1. Load audio
    waveform, sr = torchaudio.load(audio_file_path)

    # 2. Extract Mel features
    mel_features = extract_mel_features_from_waveform(
        waveform, sr,
        target_sample_rate=16000, n_mels=input_dim,
        n_fft=400, hop_length=160
    )

    # 3. Prepare for model (add batch dim, move to device, ensure float)
    input_features_for_inference = mel_features.unsqueeze(0).to(device).float()

    # 4. Perform Inference
    with torch.no_grad():
        inferred_features = transformer_model(input_features_for_inference)

    # Convert to CPU and numpy for plotting
    input_np = mel_features.cpu().numpy()
    inferred_np = inferred_features.squeeze(0).cpu().numpy()

    # 5. Create plots
    fig_mel = plt.figure(figsize=(10, 4))
    plt.imshow(input_np, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Original Mel Features (Input)')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Bins')
    plt.tight_layout()

    fig_inferred = plt.figure(figsize=(10, 4))
    plt.imshow(inferred_np.T, origin='lower', aspect='auto', cmap='magma') # Transpose for better visualization
    plt.colorbar()
    plt.title('Inferred Contextualized Features (Output)')
    plt.xlabel('Time Frames')
    plt.ylabel('Feature Dimension')
    plt.tight_layout()

    output_text = f"Input Mel Features Shape: {input_np.shape}\nInferred Contextualized Features Shape: {inferred_np.shape}"

    return output_text, fig_mel, fig_inferred