import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_views, n_global_crops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.out_dim = out_dim
        self.n_views = n_views # Total number of crops (global + local)
        self.n_global_crops = n_global_crops # Number of global crops (usually 2)
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.nepochs = nepochs
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Initialize center as a buffer
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, epoch):
        # student_output: list of n_views tensors, each of shape (B, out_dim)
        # teacher_output: list of n_global_crops tensors, each of shape (B, out_dim)

        # Detach teacher outputs as we don't want gradients flowing through the teacher
        teacher_output = [t_out.detach() for t_out in teacher_output]

        # Determine current teacher temperature
        teacher_temp = self.teacher_temp
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.warmup_teacher_temp + \
                         (self.teacher_temp - self.warmup_teacher_temp) * epoch / self.warmup_teacher_temp_epochs

        # Apply temperature scaling for student outputs (soft targets)
        student_logits = [s / self.student_temp for s in student_output]

        # Apply centering and temperature scaling for teacher outputs (sharpened targets)
        # The center is subtracted from teacher logits BEFORE softmax
        teacher_logits = [(t - self.center) / teacher_temp for t in teacher_output]

        # Compute softmax for both student and teacher distributions
        student_probs = [F.softmax(s_logit, dim=-1) for s_logit in student_logits]
        teacher_probs = [F.softmax(t_logit, dim=-1) for t_logit in teacher_logits]

        total_loss = 0
        n_loss_terms = 0

        # Compare global student crops against all global teacher crops
        for i in range(self.n_global_crops):
            for j in range(self.n_global_crops):
                if i == j: # Avoid comparing a global student crop with its corresponding global teacher crop
                    continue
                loss = torch.sum(-teacher_probs[j] * F.log_softmax(student_logits[i], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        # Compare local student crops against all global teacher crops
        for i in range(self.n_global_crops, self.n_views):
            for j in range(self.n_global_crops):
                loss = torch.sum(-teacher_probs[j] * F.log_softmax(student_logits[i], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        # Normalize the loss
        total_loss /= n_loss_terms

        # Update center with raw teacher outputs (only global crops)
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        # The center is updated using the mean of raw teacher outputs across the batch
        # teacher_output here is a list of detached tensors, only containing global crops
        if len(teacher_output) == 0:
            return
        batch_center = sum([t.mean(dim=0) for t in teacher_output]) / len(teacher_output)
        # Use the exponential moving average to update the center
        self.center = self.center * self.center_momentum + batch_center.unsqueeze(0) * (1 - self.center_momentum)
