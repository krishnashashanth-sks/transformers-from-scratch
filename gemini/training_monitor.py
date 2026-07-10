import torch

class TrainingMonitor:
    def __init__(self, log_frequency_steps=100, log_metrics=['loss', 'grad_norm'], enable_gpu_monitoring=True):
        self.log_frequency_steps = log_frequency_steps
        self.log_metrics = log_metrics
        self.history = [] # To store logged data
        self.step_counter = 0
        self.enable_gpu_monitoring = enable_gpu_monitoring

        print(f"TrainingMonitor initialized with log_frequency_steps={log_frequency_steps}, metrics={log_metrics}")

    def _log_data(self, step, data):
        """Internal method to store or print logged data."""
        self.history.append({'step': step, **data})
        # For conceptual logging, just print to console
        print(f"Step {step}: {data}")

    def log_losses(self, total_loss, individual_losses, model_output_device):
        """Logs individual and total loss values."""
        if self.step_counter % self.log_frequency_steps == 0:
            log_entry = {
                'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                'individual_losses': {k: v for k, v in individual_losses.items()}
            }
            self._log_data(self.step_counter, log_entry)

    def log_gradient_norms(self, model, clip_norm=None):
        """Logs gradient norms and clipping statistics."""
        if self.step_counter % self.log_frequency_steps == 0 and 'grad_norm' in self.log_metrics:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            log_entry = {
                'grad_norm': total_norm
            }
            if clip_norm is not None:
                # Simplified: In a real scenario, you'd check if clipping actually occurred
                log_entry['grad_clipped_ratio'] = min(1.0, total_norm / clip_norm) if total_norm > clip_norm else 0.0
            self._log_data(self.step_counter, log_entry)

    def log_resource_utilization(self):
        """Captures and logs resource utilization metrics."""
        if self.step_counter % self.log_frequency_steps == 0 and self.enable_gpu_monitoring and torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            mem_alloc = torch.cuda.memory_allocated(gpu_index) / (1024**3) # GB
            mem_cached = torch.cuda.memory_reserved(gpu_index) / (1024**3) # GB
            # Note: GPU utilization percentage often requires external tools (e.g., `nvidia-smi` via subprocess)
            # For simplicity, we'll use a placeholder or simulate a basic metric.
            # Placeholder for actual GPU utilization (e.g., from nvidia-smi output)
            gpu_utilization_percent = "N/A (requires external tool)"

            log_entry = {
                'gpu_mem_allocated_gb': mem_alloc,
                'gpu_mem_cached_gb': mem_cached,
                'gpu_util_percent': gpu_utilization_percent
            }
            self._log_data(self.step_counter, log_entry)
        elif self.step_counter % self.log_frequency_steps == 0 and not self.enable_gpu_monitoring:
            # Placeholder for CPU/RAM for non-GPU or disabled GPU monitoring
            log_entry = {
                'cpu_util_percent': 'N/A (conceptual)',
                'ram_usage_gb': 'N/A (conceptual)'
            }
            self._log_data(self.step_counter, log_entry)

    def step(self):
        self.step_counter += 1

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
