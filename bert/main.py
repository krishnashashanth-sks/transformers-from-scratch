import time
import torch.nn as nn
from model import *
from dataset import bert_dataset,pad_id,tokenize
from torch.utils.data import DataLoader
import torch.optim as optim
from train import train_step
from evaluate import evaluate_step
from inference import generate_text_from_mask

vocab_size = 30000  # Example vocabulary size
embed_dim = 768     # Typically 768 for BERT-base
max_seq_len = 512   # Maximum sequence length
num_segments = 2    # For sentence A and B
num_layers = 12     # Number of encoder layers (e.g., 12 for BERT-base)
num_heads = 12      # Number of attention heads
ff_dim = 3072       # Feed-forward dimension (4 * embed_dim)
dropout_rate = 0.1

model = BERT(vocab_size, embed_dim, max_seq_len, num_segments, num_layers, num_heads, ff_dim, dropout_rate)

vocab_size = 1000
embed_dim = 64
max_seq_len = 50
num_segments = 2
num_layers = 2
num_heads = 2
ff_dim = 128
dropout_rate = 0.1

pretraining_model = BERTForPretraining(vocab_size, embed_dim, max_seq_len, num_segments, num_layers, num_heads, ff_dim, dropout_rate)

vocab_size = 1000
embed_dim = 64
max_seq_len = 50
num_segments = 2
num_layers = 2
num_heads = 2
ff_dim = 128
dropout_rate = 0.1

mlm_model = BERTWithMLM(vocab_size, embed_dim, max_seq_len, num_segments, num_layers, num_heads, ff_dim, dropout_rate)

batch_size = 8 # Define a batch size for the DataLoader

bert_dataloader = DataLoader(
    bert_dataset,
    batch_size=batch_size,
    shuffle=True, # Shuffle the data for better training
    num_workers=0 # For simplicity, set to 0. In a real scenario, consider >0 for performance
)

bert_eval_dataloader = DataLoader(
    bert_dataset,
    batch_size=batch_size,
    shuffle=False, # Do not shuffle for evaluation
    num_workers=0
)

learning_rate = 1e-4
num_train_epochs = 20 # For a simple example, a smaller number of epochs
weight_decay = 0.01

# Confirm dropout rate from model definition or define if not present
dropout_rate_for_training = dropout_rate # Using the dropout_rate defined earlier for the model


# Instantiate CrossEntropyLoss for MLM
# We use ignore_index=pad_id so that padding tokens do not contribute to the loss
mlm_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# Instantiate CrossEntropyLoss for NSP
# For binary classification, CrossEntropyLoss is typically used without special ignore_index
nsp_criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 1. Initialize empty lists to store the training history
train_history = {
    'total_loss': [],
    'mlm_loss': [],
    'nsp_loss': []
}

print("Starting BERT pre-training...")

# 2. Start a loop that iterates num_train_epochs times
for epoch in range(num_train_epochs):
    start_time = time.time()

    # 3. Inside the epoch loop, initialize variables to accumulate the total loss, MLM loss, and NSP loss for the current epoch
    total_epoch_loss = 0
    total_epoch_mlm_loss = 0
    total_epoch_nsp_loss = 0
    num_batches = 0

    # 4. Start an inner loop that iterates through each batch in the bert_dataloader
    for i, batch in enumerate(bert_dataloader):
        # 5. Call the train_step function
        batch_total_loss, batch_mlm_loss, batch_nsp_loss = train_step(
            model, batch, mlm_criterion, nsp_criterion, optimizer, device,vocab_size
        )

        # 6. Accumulate the batch losses into the epoch's total losses
        total_epoch_loss += batch_total_loss
        total_epoch_mlm_loss += batch_mlm_loss
        total_epoch_nsp_loss += batch_nsp_loss
        num_batches += 1

    # 7. After the inner batch loop finishes, calculate the average total loss, average MLM loss, and average NSP loss for the current epoch
    avg_total_loss = total_epoch_loss / num_batches
    avg_mlm_loss = total_epoch_mlm_loss / num_batches
    avg_nsp_loss = total_epoch_nsp_loss / num_batches

    # 8. Append these average epoch losses to their respective history lists
    train_history['total_loss'].append(avg_total_loss)
    train_history['mlm_loss'].append(avg_mlm_loss)
    train_history['nsp_loss'].append(avg_nsp_loss)

    end_time = time.time()
    epoch_duration = end_time - start_time

    # 9. Print the epoch number and average losses periodically
    if (epoch + 1) % 1 == 0: # Print every epoch for detailed view
        print(f"Epoch {epoch+1}/{num_train_epochs} | Duration: {epoch_duration:.2f}s | Avg Total Loss: {avg_total_loss:.4f} | Avg MLM Loss: {avg_mlm_loss:.4f} | Avg NSP Loss: {avg_nsp_loss:.4f}")

print("BERT pre-training finished.")

print("Starting BERT evaluation...")

# 1. Initialize variables to accumulate losses
total_eval_mlm_loss = 0
total_eval_nsp_loss = 0
num_eval_batches = 0

# 2. Set the model to evaluation mode (already done in evaluate_step, but good practice)
model.eval()

# 3. Iterate through each batch in the bert_eval_dataloader
for i, batch in enumerate(bert_eval_dataloader):
    # 4. Call the evaluate_step function
    batch_mlm_loss, batch_nsp_loss = evaluate_step(
        model, batch, mlm_criterion, nsp_criterion, device
    )

    # 5. Accumulate the returned losses
    total_eval_mlm_loss += batch_mlm_loss
    total_eval_nsp_loss += batch_nsp_loss
    num_eval_batches += 1

# 6. Calculate average evaluation losses
avg_eval_mlm_loss = total_eval_mlm_loss / num_eval_batches
avg_eval_nsp_loss = total_eval_nsp_loss / num_eval_batches

# 7. Calculate average total evaluation loss
avg_eval_total_loss = avg_eval_mlm_loss + avg_eval_nsp_loss

# 8. Print the calculated average evaluation losses
print("BERT evaluation finished.")
print(f"\nEvaluation Results: Avg Total Loss: {avg_eval_total_loss:.4f} | Avg MLM Loss: {avg_eval_mlm_loss:.4f} | Avg NSP Loss: {avg_eval_nsp_loss:.4f}")

print("\n--- Testing Text Generation ---")

# Example 1: Fill in a simple masked sentence
prompt_sentence_1 = "the quick [MASK] fox jumps over the [MASK] dog"
generated_output_1 = generate_text_from_mask(
    model=model,
    tokenizer=tokenize, # Pass the tokenize function
    prompt=prompt_sentence_1,
    max_len=max_seq_len,
    device=device
)
print(f"Prompt: '{prompt_sentence_1}'")
print(f"Generated: '{generated_output_1}'")


# Example 2: Another masked sentence
prompt_sentence_2 = "water is [MASK] for all living [MASK]"
generated_output_2 = generate_text_from_mask(
    model=model,
    tokenizer=tokenize, # Pass the tokenize function
    prompt=prompt_sentence_2,
    max_len=max_seq_len,
    device=device
)
print(f"\nPrompt: '{prompt_sentence_2}'")
print(f"Generated: '{generated_output_2}'")

print("\nNote: Due to the small dataset and limited training, the generated text quality will be very low.")