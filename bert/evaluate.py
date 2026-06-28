import torch

def evaluate_step(model, batch, mlm_criterion, nsp_criterion, device,vocab_size):
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # Disable gradient calculations
        # Extract and move data to the device
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        mlm_labels = batch['mlm_labels'].to(device)
        nsp_labels = batch['nsp_labels'].squeeze(1).to(device) # Remove singleton dimension

        # Perform forward pass
        mlm_prediction_scores, nsp_prediction_scores = model(
            input_ids,
            segment_ids,
            attention_mask.unsqueeze(1).unsqueeze(2) # Unsqueeze twice for attention mask
        )

        # Calculate MLM loss
        mlm_loss = mlm_criterion(
            mlm_prediction_scores.view(-1, vocab_size), # Reshape for CrossEntropyLoss
            mlm_labels.view(-1) # Reshape for CrossEntropyLoss
        )

        # Calculate NSP loss
        nsp_loss = nsp_criterion(
            nsp_prediction_scores,
            nsp_labels
        )

    return mlm_loss.item(), nsp_loss.item()
