from tqdm.auto import tqdm
import torch

def train_model(model,num_epochs,train_dataloader,optimizer,device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} finished. Average training loss: {avg_train_loss:.4f}")
    print("Training complete!")