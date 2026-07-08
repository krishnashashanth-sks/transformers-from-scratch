import torch
from model import FlorenceScratch
import torch.nn as nn

# --- Dummy Data and Model Instantiation ---

# Model parameters (simplified for demonstration)
VOCAB_SIZE = 5000
NUM_CLASSES = 100 # For VQA, this is the number of possible answers
EMBED_DIM = 768

model = FlorenceScratch(
    vocab_size=VOCAB_SIZE,
    num_classes=NUM_CLASSES,
    visual_embed_dim=EMBED_DIM,
    lang_embed_dim=EMBED_DIM
)

# Dummy data
BATCH_SIZE = 4
IMAGE_SIZE = 224
MAX_LEN = 77

dummy_images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
dummy_text_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LEN))
dummy_labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)) # Dummy labels for VQA

print(f"Model initialized: {model.__class__.__name__}")
print(f"Dummy images shape: {dummy_images.shape}")
print(f"Dummy text tokens shape: {dummy_text_tokens.shape}")
print(f"Dummy labels shape: {dummy_labels.shape}")

# --- Optimizer and Loss Function ---

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss() # Suitable for classification tasks like VQA

print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Loss Function: {criterion.__class__.__name__}")

 # --- Training Loop (Simplified) ---

NUM_EPOCHS = 2 # Very few epochs for demonstration

print("Starting dummy training...")

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    optimizer.zero_grad() # Zero out gradients

    # Forward pass
    outputs = model(dummy_images, dummy_text_tokens)
    loss = criterion(outputs, dummy_labels)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

print("Dummy training complete.")

# Example of making a prediction after training (conceptual)
model.eval() # Set model to evaluation mode
with torch.no_grad():
    test_outputs = model(dummy_images[:1], dummy_text_tokens[:1])
    predicted_class = torch.argmax(test_outputs, dim=1)
    print(f"\nExample prediction for first dummy input: {predicted_class.item()}")