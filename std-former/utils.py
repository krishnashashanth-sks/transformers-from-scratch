import torch
import math

def q_sample(x_0, t, schedule_params):
    """Samples from q(x_t | x_0) by adding noise to x_0 at time t."""

    # Extract necessary parameters from schedule_params
    sqrt_alphas_cumprod = schedule_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = schedule_params['sqrt_one_minus_alphas_cumprod']

    # Select the correct parameters for the given timesteps t
    # Reshape for broadcasting with x_0: [batch_size, 1, 1, 1, 1]
    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(t.shape[0], 1, 1, 1, 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(t.shape[0], 1, 1, 1, 1)

    # Generate random noise epsilon
    epsilon = torch.randn_like(x_0)

    # Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * epsilon

    return x_t, epsilon # Also return epsilon, as it's often used as the training target

def p_sample(model, x_t, t, schedule_params, cond_scale=1.0):
    """Samples x_{t-1} from x_t using the model's noise prediction."""

    # 1. Ensure t is a tensor and extract relevant parameters
    t_tensor = t.long()

    # Extract parameters, ensuring they are tensors and correctly shaped for broadcasting
    betas = schedule_params['betas'].to(x_t.device)
    alphas_cumprod = schedule_params['alphas_cumprod'].to(x_t.device) # Needed for sqrt_alpha_cumprod_t
    sqrt_alphas_cumprod = schedule_params['sqrt_alphas_cumprod'].to(x_t.device)
    sqrt_one_minus_alphas_cumprod = schedule_params['sqrt_one_minus_alphas_cumprod'].to(x_t.device)
    sqrt_recip_alphas = schedule_params['sqrt_recip_alphas'].to(x_t.device)
    posterior_variance = schedule_params['posterior_variance'].to(x_t.device)

    # Select parameters for the current timestep t and reshape for broadcasting
    beta_t = betas[t_tensor].view(-1, 1, 1, 1, 1)
    # alpha_t = alphas[t_tensor].view(-1, 1, 1, 1, 1) # Not directly used in the new mean formula
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t_tensor].view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t_tensor].view(-1, 1, 1, 1, 1)
    sqrt_recip_alpha_t = sqrt_recip_alphas[t_tensor].view(-1, 1, 1, 1, 1)
    posterior_variance_t = posterior_variance[t_tensor].view(-1, 1, 1, 1, 1)

    # 2. Set model to evaluation mode
    model.eval()

    with torch.no_grad(): # Disable gradient calculations for inference
        # 3. Use the model to predict the noise epsilon_pred
        epsilon_pred = model(x_t, t_tensor)

    # 4. Calculate predicted x_0 (xstart) from epsilon_pred
    # pred_x0 = (x_t - sqrt(1 - alpha_cumprod_t) * epsilon_pred) / sqrt(alpha_cumprod_t)
    pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * epsilon_pred) / sqrt_alpha_cumprod_t

    # 5. Clamp pred_x0 to the data range (e.g., -1 to 1)
    pred_x0 = torch.clamp(pred_x0, -1., 1.)

    # 6. Re-derive epsilon from the clamped pred_x0 (epsilon_from_clamped_x0)
    # This ensures consistency with the clamping. Formula: epsilon = (x_t - sqrt(alpha_cumprod_t) * x_0) / sqrt(1 - alpha_cumprod_t)
    epsilon_from_clamped_x0 = (x_t - sqrt_alpha_cumprod_t * pred_x0) / sqrt_one_minus_alpha_cumprod_t

    # 7. Calculate the mean of the posterior distribution mu_t using epsilon_from_clamped_x0
    # As per instructions: mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * epsilon_from_clamped_x0)
    mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * epsilon_from_clamped_x0)

    # 8. Calculate the variance of the posterior distribution var_t
    var = posterior_variance_t

    # 9. If t > 0, sample a random Gaussian noise z and compute x_prev. Otherwise, x_prev = mu.
    if t_tensor.min().item() > 0:
        z = torch.randn_like(x_t)
        x_prev = mean + torch.sqrt(var) * z
    else:
        x_prev = mean

    model.train() # Set model back to training mode
    return x_prev, pred_x0

def get_cosine_schedule(num_diffusion_timesteps, s=0.008):
    """Get the cosine schedule for beta values and augment with reverse process parameters."""
    steps = torch.arange(num_diffusion_timesteps + 1, dtype=torch.float32)

    f_t = torch.cos(((steps / num_diffusion_timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    f_t = f_t / f_t[0]

    betas = 1 - (f_t[1:] / f_t[:-1])
    betas = torch.clamp(betas, 0, 0.999)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # --- New parameters for reverse process ---
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod,
        'posterior_variance': posterior_variance
    }

def p_sample_loop(model, shape, schedule_params, cond_scale=1.0, device='cuda'):
    """Samples a complete sequence from pure noise using the reverse diffusion process."""
    num_diffusion_timesteps = len(schedule_params['betas'])

    # 1. Initialize with pure Gaussian noise x_T
    x = torch.randn(shape, device=device)

    # 2. Loop through timesteps in reverse order
    for i in reversed(range(num_diffusion_timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        # 3. Call p_sample to denoise one step
        x, _ = p_sample(model, x, t, schedule_params, cond_scale)

    # 4. Return the final denoised sample x_0
    return x