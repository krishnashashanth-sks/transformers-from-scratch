from tokenizer import DummyTokenizer
from model import DBRXModel
from train import train_model
from inference import generate_text
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader,TensorDataset

def main():
    # 1. Define Hyperparameters
    vocab_size = 1000
    max_seq_len = 512
    embed_dim = 768
    num_transformer_blocks = 2
    num_heads = 12
    moe_hidden_dim = 3072
    num_experts = 4
    top_k = 2
    dropout = 0.1

    epochs = 3
    batch_size = 4
    learning_rate = 1e-4
    accumulation_steps = 2
    log_interval = 1
    checkpoint_dir = "./checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Dummy Tokenizer
    tokenizer = DummyTokenizer(vocab_size)
    pad_token_id = tokenizer.pad_token_id

    # 2. Instantiate Model, Optimizer, Loss, and Scaler
    model = DBRXModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        dropout=dropout
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id) # Ensure this is used consistently
    scaler = GradScaler() # For mixed precision

    # 3. Create Dummy DataLoaders (replace with actual data pipeline in a real scenario)
    dummy_input_ids = torch.randint(low=2, high=vocab_size, size=(100, max_seq_len), dtype=torch.long) # Avoid pad/eos
    dummy_attention_mask = torch.ones(100, max_seq_len, dtype=torch.bool)
    # Labels are shifted input_ids for language modeling
    dummy_labels = torch.cat((dummy_input_ids[:, 1:], torch.full((100, 1), pad_token_id, dtype=torch.long)), dim=1)

    train_dataset = TensorDataset(dummy_input_ids, dummy_attention_mask, dummy_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(dummy_input_ids[:20], dummy_attention_mask[:20], dummy_labels[:20]) # Smaller val set
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # 4. Implement Learning Rate Scheduler (Linear Warmup with Cosine Decay)
    # For this example, we'll use a simple CosineAnnealingLR for brevity, combined with manual warmup handling if needed.
    # In a real scenario, consider libraries like transformers for detailed schedulers.
    total_training_steps = len(train_dataloader) * epochs // accumulation_steps
    warmup_steps = int(0.1 * total_training_steps)

    # A simple linear scheduler for demonstration
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps)

    # 5. Call train_model
    print("Starting training process...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=epochs,
        accumulation_steps=accumulation_steps,
        log_interval=log_interval,
        checkpoint_dir=checkpoint_dir,
        pad_token_id=pad_token_id
    )
    print("Training process finished.")

    # After training, demonstrate text generation
    print("\nDemonstrating text generation after training...")
    prompt_text = "The quick brown fox"
    generated_text_greedy = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        device=device,
        max_new_tokens=20,
        do_sample=False
    )
    print(f"Greedy generation: {generated_text_greedy}")

    generated_text_sample = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        device=device,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )
    print(f"Sampled generation: {generated_text_sample}")


if __name__ == '__main__':
    main()