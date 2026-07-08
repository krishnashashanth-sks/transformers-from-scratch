def train_model(model, dataloader, criterion, optimizer, num_epochs, vocab_size, device, dummy_tokenizer):
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, text_tokens) in enumerate(dataloader):
            images = images.to(device)
            text_tokens = text_tokens.to(device)

            logits = model(images, text_tokens)

            targets = text_tokens[:, 1:].contiguous().view(-1)
            predictions = logits[:, :-1, :].contiguous().view(-1, vocab_size)

            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    print("Training loop finished.")
