import torch
from main import NUM_ENTITIES,NUM_RELATIONS,BATCH_SIZE,NUM_TYPES,TEXT_VOCAB_SIZE,TEXT_MAX_SEQ_LEN

def train_model(num_epochs,optimizer_advanced,model_advanced):
    for epoch in range(num_epochs):
        # Zero the gradients before running the backward pass.
        optimizer_advanced.zero_grad()

        # Generate dummy data for each batch (in a real scenario, this would come from a data loader)
        positive_heads = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))
        positive_relations = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,))
        positive_tails = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))
        negative_tails = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))
        head_types = torch.randint(0, NUM_TYPES, (BATCH_SIZE,))
        tail_types = torch.randint(0, NUM_TYPES, (BATCH_SIZE,))
        head_text = torch.randint(1, TEXT_VOCAB_SIZE, (BATCH_SIZE, TEXT_MAX_SEQ_LEN))
        tail_text = torch.randint(1, TEXT_VOCAB_SIZE, (BATCH_SIZE, TEXT_MAX_SEQ_LEN))
        relation_text = torch.randint(1, TEXT_VOCAB_SIZE, (BATCH_SIZE, TEXT_MAX_SEQ_LEN))

        # Forward pass to get scores for positive and negative samples
        positive_scores = model_advanced(positive_heads, positive_relations, positive_tails,
                                        head_types=head_types, tail_types=tail_types,
                                        head_text_indices=head_text, tail_text_indices=tail_text,
                                        relation_text_indices=relation_text)

        negative_scores = model_advanced(positive_heads, positive_relations, negative_tails,
                                        head_types=head_types, tail_types=tail_types,
                                        head_text_indices=head_text, tail_text_indices=tail_text,
                                        relation_text_indices=relation_text)

        # Calculate loss
        loss = model_advanced.loss_function(positive_scores, negative_scores)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step
        optimizer_advanced.step()

        if (epoch + 1) % 1 == 0: # Print loss every epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Training with dummy data complete.")
    print(f"Final Loss: {loss.item():.4f}")