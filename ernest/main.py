import torch.optim as optim
from model import ERNESTAdvanced
from train import train_model
import torch

# Define hyperparameters and sizes
NUM_ENTITIES = 1000
NUM_RELATIONS = 100
NUM_TYPES = 50
EMBEDDING_DIM = 128 # d_model for transformer
TEXT_VOCAB_SIZE = 5000  # Example vocabulary size for text
TEXT_MAX_SEQ_LEN = 50   # Max sequence length for Transformer encoder
BATCH_SIZE = 64

# Transformer specific hyperparameters
TEXT_ENCODER_NUM_HEADS = 4
TEXT_ENCODER_NUM_LAYERS = 2
TEXT_ENCODER_D_FF_RATIO = 4 # d_ff = embedding_dim * ratio
TEXT_ENCODER_DROPOUT = 0.1 # Added dropout rate

# Initialize the advanced model
model_advanced = ERNESTAdvanced(NUM_ENTITIES, NUM_RELATIONS, NUM_TYPES, EMBEDDING_DIM,
                                text_vocab_size=TEXT_VOCAB_SIZE,
                                text_encoder_num_heads=TEXT_ENCODER_NUM_HEADS,
                                text_encoder_num_layers=TEXT_ENCODER_NUM_LAYERS,
                                text_encoder_d_ff_ratio=TEXT_ENCODER_D_FF_RATIO,
                                text_max_seq_len=TEXT_MAX_SEQ_LEN,
                                text_encoder_dropout=TEXT_ENCODER_DROPOUT) # Pass dropout

# Dummy data for demonstration
# In a real scenario, you'd load your knowledge graph triples, entity/relation types, and text descriptions
positive_heads = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))
positive_relations = torch.randint(0, NUM_RELATIONS, (BATCH_SIZE,))
positive_tails = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))

# Negative sampling: replace tails for negative examples
negative_tails = torch.randint(0, NUM_ENTITIES, (BATCH_SIZE,))

# Dummy type data
head_types = torch.randint(0, NUM_TYPES, (BATCH_SIZE,))
tail_types = torch.randint(0, NUM_TYPES, (BATCH_SIZE,))

# Dummy text data (sequences of token IDs) - now respecting TEXT_MAX_SEQ_LEN
head_text = torch.randint(1, TEXT_VOCAB_SIZE, (BATCH_SIZE, TEXT_MAX_SEQ_LEN)) # 0 is usually padding ID
tail_text = torch.randint(1, TEXT_VOCAB_SIZE, (BATCH_SIZE, TEXT_MAX_SEQ_LEN))
relation_text = torch.randint(1, TEXT_VOCAB_SIZE, (BATCH_SIZE, TEXT_MAX_SEQ_LEN))

# Forward pass
positive_scores_advanced = model_advanced(positive_heads, positive_relations, positive_tails,
                                        head_types=head_types, tail_types=tail_types,
                                        head_text_indices=head_text, tail_text_indices=tail_text,
                                        relation_text_indices=relation_text)

negative_scores_advanced = model_advanced(positive_heads, positive_relations, negative_tails,
                                        head_types=head_types, tail_types=tail_types,
                                        head_text_indices=head_text, tail_text_indices=tail_text,
                                        relation_text_indices=relation_text)

# Calculate loss
loss_advanced = model_advanced.loss_function(positive_scores_advanced, negative_scores_advanced)
print(f"Initial Loss (Advanced ERNEST): {loss_advanced.item():.4f}")

# Optimizer (e.g., Adam)
optimizer_advanced = optim.Adam(model_advanced.parameters(), lr=0.001)

num_epochs = 10 # Define number of training epochs
train_model(num_epochs,optimizer_advanced,model_advanced)

# Generate new dummy data for inference
# In a real scenario, this would be your test set or new triples to predict
inference_batch_size = 5 # Smaller batch size for demonstration
inference_heads = torch.randint(0, NUM_ENTITIES, (inference_batch_size,))
inference_relations = torch.randint(0, NUM_RELATIONS, (inference_batch_size,))
inference_tails = torch.randint(0, NUM_ENTITIES, (inference_batch_size,))

inference_head_types = torch.randint(0, NUM_TYPES, (inference_batch_size,))
inference_tail_types = torch.randint(0, NUM_TYPES, (inference_batch_size,))

inference_head_text = torch.randint(1, TEXT_VOCAB_SIZE, (inference_batch_size, TEXT_MAX_SEQ_LEN))
inference_tail_text = torch.randint(1, TEXT_VOCAB_SIZE, (inference_batch_size, TEXT_MAX_SEQ_LEN))
inference_relation_text = torch.randint(1, TEXT_VOCAB_SIZE, (inference_batch_size, TEXT_MAX_SEQ_LEN))

# Ensure model is in evaluation mode
model_advanced.eval()

with torch.no_grad(): # Disable gradient calculations for inference
    inference_scores = model_advanced(inference_heads, inference_relations, inference_tails,
                                      head_types=inference_head_types, tail_types=inference_tail_types,
                                      head_text_indices=inference_head_text,
                                      tail_text_indices=inference_tail_text,
                                      relation_text_indices=inference_relation_text)

print("Inference complete.")
print("\nPredicted scores for new triples:")
for i, score in enumerate(inference_scores):
    print(f"Triple {i+1} Score: {score.item():.4f}")
