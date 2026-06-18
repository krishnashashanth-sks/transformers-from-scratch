def train(num_epochs,dataloader,model,optimizer,criterion,vocab_size,device):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Calculate loss
            # For next token prediction, we need to shift the labels and flatten the outputs
            # outputs shape: (batch_size, seq_len, vocab_size)
            # labels shape: (batch_size, seq_len)

            # Flatten outputs: (batch_size * seq_len, vocab_size)
            # Flatten labels: (batch_size * seq_len)
            loss = criterion(outputs[:, :-1, :].reshape(-1, vocab_size), labels[:, 1:].reshape(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {avg_loss:.4f}\n")
