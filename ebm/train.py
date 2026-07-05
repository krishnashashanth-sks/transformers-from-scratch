import torch.optim as optim
import torch
from sampler import langevin_sampler

# --- Training Loop with Persistent Contrastive Divergence (PCD) ---
def train_ebm_pcd(energy_model, dataloader, epochs=100, lr=1e-4, pcd_buffer_size=1000, pcd_refresh_rate=0.05):
    optimizer = optim.Adam(energy_model.parameters(), lr=lr)

    # Determine the device of the model to ensure data is on the same device
    device = next(energy_model.parameters()).device

    # Determine the shape of the data from the dataloader
    data_sample_shape = None
    for batch in dataloader:
        # Dataloader yields a tuple (data, labels) or just (data,) if only features
        sample_data = batch[0] if isinstance(batch, (list, tuple)) else batch
        data_sample_shape = sample_data.shape[1:] # Get shape without batch dim (e.g., (C, H, W) or (input_dim,))
        break
    if data_sample_shape is None:
        raise ValueError("Could not infer data sample shape from dataloader.")

    # PCD: Initialize negative sample buffer outside the training loop
    # This buffer stores persistent negative samples across iterations, with the correct data_sample_shape.
    negative_sample_buffer = torch.randn((pcd_buffer_size, *data_sample_shape)).to(device) # Initialize on device

    print(f"Starting EBM training with PCD on device: {device}")
    print(f"Inferred data sample shape: {data_sample_shape}")

    for epoch in range(epochs):
        for i, data_batch_tuple in enumerate(dataloader):
            # Dataloader yields a tuple (data, labels) or just (data,) if only features
            real_data = data_batch_tuple[0] if isinstance(data_batch_tuple, (list, tuple)) else data_batch_tuple
            real_data = real_data.to(device) # Move real data to the model's device

            batch_size = real_data.shape[0]

            # PCD: Prepare initial negative samples for Langevin Dynamics
            # A portion comes from the persistent buffer, and some from fresh noise for exploration.
            num_from_buffer = int(batch_size * (1 - pcd_refresh_rate))
            num_from_noise = batch_size - num_from_buffer

            # Randomly select samples from the persistent buffer
            initial_negative_samples_from_buffer = torch.empty((0, *data_sample_shape), device=device) # Initialize empty tensor
            if num_from_buffer > 0 and negative_sample_buffer.shape[0] >= num_from_buffer:
                idx_from_buffer = torch.randperm(negative_sample_buffer.shape[0])[:num_from_buffer]
                initial_negative_samples_from_buffer = negative_sample_buffer[idx_from_buffer].to(device)
            elif num_from_buffer > 0: # Buffer is too small or empty, generate noise instead
                print("Warning: PCD buffer insufficient, generating more noise for buffer samples.")
                num_from_noise += num_from_buffer # Add to noise generation
                num_from_buffer = 0

            # Generate fresh random noise samples, using the determined data_sample_shape
            initial_negative_samples_from_noise = torch.randn((num_from_noise, *data_sample_shape), device=device)

            # Combine them to form the initial samples for Langevin
            initial_negative_samples = torch.cat(
                [initial_negative_samples_from_buffer, initial_negative_samples_from_noise], dim=0
            )
            # Shuffle to mix them thoroughly
            initial_negative_samples = initial_negative_samples[torch.randperm(initial_negative_samples.shape[0])]

            # 1. Generate negative samples using MCMC (e.g., Langevin Dynamics)
            negative_samples = langevin_sampler(energy_model, initial_negative_samples)

            # PCD: Update the negative sample buffer with the newly generated samples
            # Replace an equal number of old samples in the buffer with the new ones.
            if pcd_buffer_size >= batch_size:
                # Select random indices in the buffer to be replaced
                replace_idx = torch.randperm(pcd_buffer_size)[:batch_size]
                # Detach and move to CPU to avoid memory issues if buffer is large and GPU memory is limited
                # Or keep on device if GPU memory permits, but for generality, keep it on CPU for buffer.
                negative_sample_buffer[replace_idx] = negative_samples.detach() # Keep on device, or move to CPU if buffer was on CPU
            else:
                # If buffer is smaller than batch_size, handle as needed, e.g., reinitialize or resize
                # For simplicity here, we'll just reinitialize for this edge case.
                negative_sample_buffer = negative_samples.detach() # Keep on device
                print("Warning: PCD buffer too small, reinitialized with current batch.")

            # 2. Calculate energies for real and negative samples
            energy_real = energy_model(real_data)
            energy_negative = energy_model(negative_samples)

            # 3. Compute the Contrastive Divergence loss
            # Minimize E(real_data) and maximize E(negative_samples)
            loss = (energy_real - energy_negative).mean()

            # 4. Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
