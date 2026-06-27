import torch

def train(n_epochs,train_loader,student_vit,student_head,teacher_vit,teacher_head,dino_loss,optimizer,n_global_crops,batch_size,center_momentum,device):
    print("Starting training loop...")
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_idx, views in enumerate(train_loader):
            # Move all views to device
            views = [v.to(device) for v in views]

            # Process each view individually through the student network
            student_output_views = []
            for view in views:
                s_vit_out = student_vit(view)
                s_head_out = student_head(s_vit_out)
                student_output_views.append(s_head_out)

            # Separate global crops for teacher (teacher only processes global crops)
            teacher_input = torch.cat(views[:n_global_crops], dim=0)

            # Teacher forward pass (detached)
            with torch.no_grad():
                teacher_output_vit = teacher_vit(teacher_input)
                teacher_output = teacher_head(teacher_output_vit)

            # Split teacher output back into views (since teacher_input was concatenated)
            teacher_output_views = torch.split(teacher_output, batch_size)

            # Calculate DINO loss
            loss = dino_loss(student_output_views, teacher_output_views, epoch)

            # Zero gradients, backprop, and update student weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher network's weights using EMA
            with torch.no_grad():
                m = center_momentum # The momentum coefficient (fixed for now, but often ramps up)
                for param_s, param_t in zip(student_vit.parameters(), teacher_vit.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.data)
                for param_s, param_t in zip(student_head.parameters(), teacher_head.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.data)

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

    print("Training loop finished.")