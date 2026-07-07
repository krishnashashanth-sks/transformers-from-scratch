def train_model(num_epochs,train_data_loader,model,gen_loss_fct,disc_loss_fct,gen_optimizer,disc_optimizer,vocab_size):
    # Set model to training mode
    model.train()

    print("Starting training loop with real dataset...")

    for epoch in range(num_epochs):
        total_gen_loss = 0
        total_disc_loss = 0
        num_batches = 0

        for batch in train_data_loader: # Changed from dummy_data_loader to train_data_loader
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            original_labels_for_mlm = batch['original_labels']

            # --- Forward pass for ELECTRA ---
            gen_logits, gen_mlm_labels, disc_logits, disc_labels = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                original_labels=original_labels_for_mlm
            )

            # --- Calculate Generator Loss (MLM Loss) ---
            gen_loss = gen_loss_fct(gen_logits.view(-1, vocab_size), gen_mlm_labels.view(-1))

            # --- Calculate Discriminator Loss ---
            active_discriminator_loss = (attention_mask == 1).view(-1)
            # Ensure disc_logits and disc_labels are flattened and only non-padding elements are considered
            disc_loss = disc_loss_fct(
                disc_logits.view(-1)[active_discriminator_loss],
                disc_labels.view(-1)[active_discriminator_loss]
            )

            # --- Backpropagation (Alternating or Joint) ---
            # Generator training step
            gen_optimizer.zero_grad()
            gen_loss.backward(retain_graph=True) # retain_graph=True if discriminator uses generator outputs graph
            gen_optimizer.step()

            # Discriminator training step
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            num_batches += 1

        avg_gen_loss = total_gen_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches

        print(f"Epoch {epoch + 1}/{num_epochs} - Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")

    print("Training complete!")