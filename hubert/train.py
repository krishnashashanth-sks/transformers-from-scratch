from sklearn.cluster import MiniBatchKMeans
import torch
import tqdm
from utils import apply_span_masking
import matplotlib.pyplot as plt
import numpy as np

def train(num_iterations,transformer_model,prediction_head,epochs_per_iteration,hubert_dataloader_with_targets,temp_dataloader_eval,hubert_dataset,num_clusters,optimizer,hubert_loss,device):
    for iteration in range(num_iterations):
        print(f"\n--- HuBERT Iteration {iteration+1}/{num_iterations} ---")

        # Phase 1: Train Masked Prediction Model
        transformer_model.train()
        prediction_head.train()
        print(f"Training for {epochs_per_iteration} epochs with current discrete targets.")

        for epoch in range(epochs_per_iteration):
            total_loss = 0
            num_batches = 0
            # Wrap the dataloader with tqdm for a progress bar
            for batch_idx, (mel_features_batch, targets_batch, lengths_batch) in enumerate(tqdm.tqdm(hubert_dataloader_with_targets, desc=f"Epoch {epoch+1}")):
                optimizer.zero_grad()

                # Move batch data to device
                mel_features_batch = mel_features_batch.to(device)
                targets_batch = targets_batch.to(device)

                # Apply masking to the full features
                # Note: We use the *original* features for masking, but predict current discrete_targets.
                # The features are (batch_size, n_mels, time_steps)
                masked_features, mask_indices = apply_span_masking(
                    mel_features_batch.float(),
                    mask_prob=0.15,
                    mask_length=10,
                    special_mask_token=torch.tensor(-10.0, device=device) # Use a scalar tensor for broadcasting
                )

                # Forward pass
                # transformer_output shape: (batch_size, time_steps, model_dim)
                transformer_output = transformer_model(masked_features)
                # prediction_logits shape: (batch_size, time_steps, num_clusters)
                prediction_logits = prediction_head(transformer_output)

                # Calculate loss using the current (possibly re-clustered) discrete_targets
                # Note: targets_batch contains padding with -1. hubert_loss handles this implicitly
                # by only considering masked_targets which are extracted using mask_indices.
                # Ensure targets_batch is flattened and on CPU for hubert_loss
                loss = hubert_loss(prediction_logits, targets_batch.cpu().numpy().flatten(), mask_indices)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1}/{epochs_per_iteration}, Average Loss: {avg_loss:.4f}")

        # Phase 2: Extract Contextualized Features for Re-clustering
        print("Extracting features for re-clustering...")
        transformer_model.eval() # Set model to evaluation mode

        all_extracted_features = []
        all_original_lengths_for_recluster = [] # Need to re-collect lengths as dataloader order changes

        # Use a temporary dataloader with shuffle=False to ensure consistent order for feature extraction
        

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

                # if (batch_idx + 1) % 1000 == 0:
                #     print(f"  Extracted features for {batch_idx + 1}/{len(hubert_dataset)} audio files.")

        # Concatenate all extracted features for clustering
        features_for_re_clustering = torch.cat(all_extracted_features, dim=0).cpu().float().numpy()
        print(f"Total frames for re-clustering: {features_for_re_clustering.shape[0]}")

        # Phase 3: Re-cluster Features to Generate New Targets
        print("Re-clustering features to generate new discrete targets...")
        kmeans_re_model = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, n_init=10, batch_size=256, verbose=False)
        kmeans_re_model.fit(features_for_re_clustering)

        # Update the global discrete_targets for the next iteration by assigning back to dataset
        new_discrete_targets = kmeans_re_model.labels_
        hubert_dataset.assign_discrete_targets(new_discrete_targets, all_original_lengths_for_recluster)
        print(f"New discrete targets generated. Unique clusters: {len(np.unique(new_discrete_targets))}")

        # Visualize new cluster distribution (optional, for debugging/understanding)
        plt.figure(figsize=(10, 4))
        plt.hist(new_discrete_targets, bins=np.arange(num_clusters + 1) - 0.5, rwidth=0.8, density=True)
        plt.title(f'Distribution of Discrete Targets (Iteration {iteration+1})')
        plt.xlabel('Cluster ID')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
