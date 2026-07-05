
import torch

def train_model(num_epochs,model,dataloader,optimizer,noise_prediction_loss_fn,batch_size,log_interval,device):
    print("Starting training loop...")
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # 2. Move inputs to the appropriate device
            tokenized_text = batch['tokenized_text'].to(device)
            ref_image = batch['ref_image'].to(device)
            dummy_nerf_latent = batch['dummy_nerf_latent'].to(device)

            # 3a. Zero the optimizer's gradients
            optimizer.zero_grad()

            # 3b. Generate a random timestep for the diffusion process
            # The number of diffusion steps is implicitly defined in ConditionalDiffusionModel.betas
            T = len(model.conditional_diffusion_model.betas) # Total diffusion steps (e.g., 1000)
            timesteps = torch.randint(0, T, (batch_size,), device=device).long()

            # The diffusion model aims to predict the noise, so we need actual noise
            # to compare against for the loss.
            # This 'noise' is applied to the dummy_nerf_latent to create x_t.
            actual_noise = torch.randn_like(dummy_nerf_latent.unsqueeze(-1)).to(device)

            # 3c. Use the model to predict noise
            # The model's forward method is designed for training when target_neRF_latent_for_diffusion is provided.
            # The `dummy_nerf_latent` acts as the `x_start` in the diffusion process, to which noise is added.
            # The model then predicts `actual_noise` given the noisy latent and conditions.
            predicted_noise = model(
                text_input=tokenized_text,
                image_input=ref_image,
                timesteps=timesteps,
                target_neRF_latent_for_diffusion=dummy_nerf_latent, # This is x_start
            )

            # 3d. Calculate the noise_prediction_loss
            # We need to compute the noise added to `dummy_nerf_latent` at `timesteps`
            # The `q_sample` method does this by adding `actual_noise` to `dummy_nerf_latent`.
            # The `predicted_noise` is what the model thinks that `actual_noise` was.
            loss = noise_prediction_loss_fn(predicted_noise, actual_noise)

            # 3e. Perform backpropagation and update model parameters
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {avg_loss:.4f}')

    print("Training loop finished.")
