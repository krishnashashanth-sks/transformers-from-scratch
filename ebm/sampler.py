import torch
from torch.autograd import grad

# --- MCMC Sampler (Langevin Dynamics) ---
def langevin_sampler(energy_model, initial_samples, n_steps=100, step_size=0.1, noise_scale=0.005):
    # Clone samples to avoid modifying in-place and ensure they require grad for energy_grad computation
    samples = initial_samples.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        # Calculate energy for current samples
        energies = energy_model(samples)

        # Compute gradients of energy w.r.t. samples (this is crucial!)
        # We need to sum gradients over batch elements as energy_model outputs a scalar for each sample
        energy_grad = grad(energies.sum(), samples, create_graph=True)[0]

        # Langevin update rule
        samples.data -= step_size * energy_grad.data  # Move towards lower energy
        samples.data += noise_scale * torch.randn_like(samples).data  # Add noise

    return samples.detach()
