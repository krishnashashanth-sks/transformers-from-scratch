import torch
import torch.nn as nn
import os
from model import FlamingoLikeModel
from utils import setup_data
from train import train_model
from evaluate import evaluate_model
from inference import generate_caption

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration Parameters ---
    # Model dimensions and parameters
    embed_dim = 256 # Embedding dimension for all modules
    num_heads = 8
    dimm_head = embed_dim // num_heads # Ensure dim_head * num_heads = embed_dim
    ff_hidden_mult = 4
    dropout = 0.1

    # Vision Encoder parameters
    img_size = 64
    patch_size = 8
    in_channels = 3
    vision_depth = 3 # Number of transformer encoder blocks

    # Perceiver Resampler parameters
    perceiver_num_latents = 8 # Number of visual tokens to output
    perceiver_num_cross_attention_layers = 1
    perceiver_num_self_attention_layers = 2

    # Language Model parameters
    MAX_SEQ_LEN = 20 # Max sequence length for the LM (including visual tokens)
    language_num_decoder_blocks = 3

    # Training parameters
    num_epochs = 5
    learning_rate = 1e-4
    BATCH_SIZE = 2
    NUM_VISUAL_TOKENS = perceiver_num_latents # Must match perceiver_num_latents

    print("Configuration parameters defined.")

    # --- Data Setup ---
    dummy_image_dir = "./dummy_images"
    image_paths = [
        os.path.join(dummy_image_dir, f"image{i}.png") for i in range(3)
    ]
    dummy_data = [
        (image_paths[0], "Hello world, this is a test image caption."),
        (image_paths[1], "Another example of multimodal data."),
        (image_paths[2], "This is the third image with a caption."),
    ]

    dummy_tokenizer, dummy_image_transform, multimodal_dataset, multimodal_dataloader = setup_data(
        dummy_image_dir, image_paths, dummy_data, MAX_SEQ_LEN, NUM_VISUAL_TOKENS, BATCH_SIZE
    )
    vocab_size = dummy_tokenizer.vocab_size
    print("Dummy data and DataLoader set up.")

    # --- Model Instantiation ---
    model = FlamingoLikeModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        vision_embed_dim=embed_dim,
        vision_depth=vision_depth,
        vision_num_heads=num_heads,
        vision_dim_head=dimm_head,
        vision_dropout=dropout,
        vision_ff_hidden_mult=ff_hidden_mult,
        perceiver_num_latents=perceiver_num_latents,
        perceiver_latent_dim=embed_dim,
        perceiver_num_cross_attention_heads=num_heads,
        perceiver_num_self_attention_heads=num_heads,
        perceiver_num_cross_attention_layers=perceiver_num_cross_attention_layers,
        perceiver_num_self_attention_layers=perceiver_num_self_attention_layers,
        perceiver_cross_attention_dropout=dropout,
        perceiver_self_attention_dropout=dropout,
        perceiver_ff_dropout=dropout,
        perceiver_ff_hidden_mult=ff_hidden_mult,
        perceiver_dim_head=dimm_head,
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        language_dim=embed_dim,
        language_num_decoder_blocks=language_num_decoder_blocks,
        language_num_heads=num_heads,
        language_dim_head=dimm_head,
        language_dropout=dropout,
        language_ff_hidden_mult=ff_hidden_mult
    ).to(device)
    print("FlamingoLikeModel instantiated and moved to device.")

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss(ignore_index=dummy_tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Loss function and optimizer defined.")

    # --- Training ---
    train_model(model, multimodal_dataloader, criterion, optimizer, num_epochs, vocab_size, device, dummy_tokenizer)

    # --- Evaluation ---
    evaluate_model(model, multimodal_dataloader, criterion, vocab_size, device, dummy_tokenizer)

    # --- Inference ---
    print("\nStarting inference...")
    sample_image, _ = multimodal_dataset[0] # Get image from the first sample
    generated_caption = generate_caption(
        model=model,
        image=sample_image,
        tokenizer=dummy_tokenizer,
        max_new_tokens=10,
        max_seq_len=MAX_SEQ_LEN,
        num_visual_tokens=NUM_VISUAL_TOKENS,
        device=device,
        start_text="This is a"
    )
    print(f"Generated Caption: {generated_caption}")

    # --- Cleanup --- #
    for img_path in image_paths:
        if os.path.exists(img_path):
            os.remove(img_path)
    if os.path.exists(dummy_image_dir):
        os.rmdir(dummy_image_dir)
    print("Dummy image files and directory cleaned up.")
    print("End of full demonstration.")
