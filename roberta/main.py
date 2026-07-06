import torch
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch
from model import RoBERTaForSequenceClassification
from config import RoBERTaConfig
from train import train_step
from evaluate import eval_step
from inference import predict

config = RoBERTaConfig()
num_labels = 2 # Example for binary classification
roberta_sequence_classifier = RoBERTaForSequenceClassification(config, num_labels)
print(roberta_sequence_classifier)

# Define batch size and sequence length for dummy inputs
batch_size = 4
sequence_length = 128

# 1. Generate dummy input_ids
dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length), dtype=torch.long)

# 2. Generate dummy attention_mask
dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)

# 3. Generate dummy labels
dummy_labels = torch.randint(0, num_labels, (batch_size,), dtype=torch.long)

print(f"Dummy input_ids shape: {dummy_input_ids.shape}")
print(f"Dummy attention_mask shape: {dummy_attention_mask.shape}")
print(f"Dummy labels shape: {dummy_labels.shape}")

# Initialize AdamW optimizer
learning_rate = 5e-5
optimizer = AdamW(roberta_sequence_classifier.parameters(), lr=learning_rate)
print(f"AdamW optimizer initialized with learning rate: {learning_rate}")

# Initialize StepLR learning rate scheduler
step_size = 1
gamma = 0.9
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
print(f"StepLR scheduler initialized with step_size: {step_size} and gamma: {gamma}")

num_epochs = 3 # Define the number of epochs

print("Starting training loop...")

for epoch in range(num_epochs):
    # Training step
    train_loss, train_logits = train_step(
        roberta_sequence_classifier,
        optimizer,
        dummy_input_ids,
        dummy_attention_mask,
        dummy_labels
    )

    # Update learning rate scheduler
    scheduler.step()

    # Evaluation step (using dummy data as validation set for simplicity)
    eval_loss, eval_accuracy = eval_step(
        roberta_sequence_classifier,
        dummy_input_ids,
        dummy_attention_mask,
        dummy_labels
    )

    print(f"Epoch {epoch + 1}/{num_epochs}: ")
    print(f"  Training Loss: {train_loss:.4f}")
    print(f"  Evaluation Loss: {eval_loss:.4f}")
    print(f"  Evaluation Accuracy: {eval_accuracy:.4f}")

print("Training loop finished.")

# Perform prediction using the defined predict function
predicted_labels = predict(
    roberta_sequence_classifier,
    dummy_input_ids,
    dummy_attention_mask
)

print(f"Predicted labels shape: {predicted_labels.shape}")
print(f"Predicted labels: {predicted_labels}")