# Define the training function
def train_fsd_system(model, train_dataloader, val_dataloader, loss_calculator,
                     optimizer, scheduler, scaler, device, num_epochs, max_grad_norm,
                     params, checkpoint_dir="checkpoints"):

    print("\n--- Starting FSDSystem Training Process ---")
    best_val_metric = -float('inf') # Initialize for 'higher is better' metric

    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        
        # Training Loop
        for batch_idx, (model_inputs, ground_truths) in enumerate(train_dataloader):
            optimizer.zero_grad()  # Zero gradients for each batch

            # Move inputs and ground truths to the appropriate device
            # Note: This is simplified. In a real scenario, a custom collate_fn
            # in DataLoader would handle moving tensors to device.
            for key in model_inputs:
                if isinstance(model_inputs[key], torch.Tensor):
                    model_inputs[key] = model_inputs[key].to(device)
            for key in ground_truths:
                if isinstance(ground_truths[key], torch.Tensor):
                    ground_truths[key] = ground_truths[key].to(device)

            # Mixed-precision training: forward pass with autocast
            with torch.cuda.amp.autocast():
                model_outputs = model(
                    cam_input_sequence=model_inputs['cam_input_sequence'],
                    lidar_input_sequence=model_inputs['lidar_input_sequence'],
                    radar_input_sequence=model_inputs['radar_input_sequence'],
                    occupancy_query_points=model_inputs['occupancy_query_points'],
                    detected_agents_states_seq=model_inputs['detected_agents_states_seq'],
                    ego_vehicle_state=model_inputs['ego_vehicle_state']
                )
                loss_info = loss_calculator(model_outputs, ground_truths)
                loss = loss_info['total_loss']
            
            # Backward pass and optimizer step with GradScaler
            scaler.scale(loss).backward()  # Scale loss and compute gradients
            scaler.unscale_(optimizer)     # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm) # Gradient clipping
            scaler.step(optimizer)         # Update model weights
            scaler.update()                # Update the GradScaler for the next iteration

            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0: # Print every 10 batches
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"--- Epoch {epoch+1}/{num_epochs} Training Complete. Avg Train Loss: {avg_train_loss:.4f} ---")

        # Evaluation Loop
        avg_val_loss, _, best_val_metric = evaluate(
            model, val_dataloader, loss_calculator, params, device, epoch + 1, best_val_metric, checkpoint_dir
        )
        
        scheduler.step() # Step learning rate scheduler after each epoch (or customize to step after each batch)

    print("\n--- FSDSystem Training Process Complete ---")
    return model, best_val_metric