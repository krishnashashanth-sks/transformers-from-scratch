import torch
from utils import q_sample

def train_model(epochs,train_dataloader,num_diffusion_timesteps,model,optimizer,loss_fn,schedule_params,device):
    print("\nStarting training loop...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss = 0
        for step, (x_0, _) in enumerate(train_dataloader):
            optimizer.zero_grad()

            x_0 = x_0.to(device) # Move clean data to device

            # Sample a random timestep for each item in the batch
            t = torch.randint(0, num_diffusion_timesteps, (x_0.shape[0],), device=device).long()

            # Add noise to x_0 to get x_t and the true noise epsilon
            x_t, epsilon = q_sample(x_0, t, schedule_params)

            # Predict noise using the STDiT model
            epsilon_pred = model(x_t, t)

            # Calculate loss (MSE between predicted and true noise)
            loss = loss_fn(epsilon_pred, epsilon)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 5 == 0: # Log every few steps
                print(f"  Epoch {epoch+1}/{epochs}, Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}\n")

    print("Training loop finished.")