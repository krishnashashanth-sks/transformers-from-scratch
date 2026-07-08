import torch
from layers import VisionTransformer
from tokenizer import BBoxTokenizer
from utils import dummy_text_tokenizer
from model import MultimodalTextTransformer

# --- Bounding Box Tokenization Configuration (from Step 4) ---
NUM_COORD_BINS = 1000
BASE_VOCAB_SIZE = 30000
NEW_VOCAB_SIZE = BASE_VOCAB_SIZE + NUM_COORD_BINS

# --- Inference Example ---
print('--- Starting Inference Example ---')

# Define common parameters
image_size = 256
patch_size = 32
in_channels = 3
embed_dim = 768
depth = 6
heads = 12
mlp_dim = 2048
dropout = 0.1
emb_dropout = 0.1
max_seq_len = 128 # Max sequence length for text tokens + bbox tokens

# Instantiate Vision Encoder
vision_encoder = VisionTransformer(
    image_size = image_size,
    patch_size = patch_size,
    in_channels = in_channels,
    num_classes = 0, # To get raw visual features
    embed_dim = embed_dim,
    depth = depth,
    heads = heads,
    mlp_dim = mlp_dim,
    dropout = dropout,
    emb_dropout = emb_dropout
)
vision_encoder.eval()

# Instantiate Bounding Box Tokenizer
bbox_tokenizer = BBoxTokenizer(num_coord_bins=NUM_COORD_BINS, coord_token_offset=BASE_VOCAB_SIZE)

# Prepare dummy text with an explicit bounding box reference
raw_text = "A cat sits on this mat. The cat is inside the box."
dummy_bbox_coords = [0.1, 0.2, 0.5, 0.6] # Normalized coordinates for "the box"

# Tokenize text and integrate bounding box tokens
input_token_ids = dummy_text_tokenizer(raw_text)
bbox_token_ids = bbox_tokenizer.tokenize_bbox(dummy_bbox_coords)

# Conceptually insert bbox tokens. In a real system, placement is semantic.
# For this example, we'll append it to represent a description including a box.
combined_tokens = input_token_ids + bbox_token_ids

# Pad or truncate to max_seq_len
if len(combined_tokens) > max_seq_len:
    combined_tokens = combined_tokens[:max_seq_len]
else:
    combined_tokens = combined_tokens + [0] * (max_seq_len - len(combined_tokens))

dummy_text_input = torch.tensor(combined_tokens, dtype=torch.long).unsqueeze(0) # Add batch dimension

# Create a dummy image tensor
dummy_image = torch.randn(1, in_channels, image_size, image_size)

# Instantiate Multimodal Text Transformer with the NEW_VOCAB_SIZE
multimodal_text_transformer = MultimodalTextTransformer(
    vocab_size = NEW_VOCAB_SIZE,
    max_seq_len = max_seq_len,
    embed_dim = embed_dim,
    depth = depth,
    heads = heads,
    mlp_dim = mlp_dim,
    dropout = dropout,
    emb_dropout = emb_dropout
)
multimodal_text_transformer.eval()

print(f"Dummy image shape: {dummy_image.shape}")
print(f"Dummy tokenized text input shape: {dummy_text_input.shape}")
print(f"Example combined tokens (words + bbox): {dummy_text_input[0, :len(input_token_ids) + len(bbox_token_ids)]}")
print(f"Descriptive bbox tokens: {bbox_tokenizer.describe_bbox_tokens(bbox_token_ids)}")

with torch.no_grad():
    # Forward pass through vision encoder
    visual_tokens = vision_encoder(dummy_image)

    # Forward pass through multimodal text transformer
    multimodal_logits = multimodal_text_transformer(dummy_text_input, visual_tokens)

print(f"Visual tokens shape from Vision Encoder: {visual_tokens.shape}")
print(f"Output multimodal logits shape: {multimodal_logits.shape}")

# To get the predicted next token (conceptual)
predicted_next_token_logits = multimodal_logits[0, -1, :]
predicted_token_id = torch.argmax(predicted_next_token_logits).item()

print(f"Predicted next token ID (from last position): {predicted_token_id}")
print("--- Inference Example Finished ---")