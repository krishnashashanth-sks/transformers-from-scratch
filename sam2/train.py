from tqdm.auto import tqdm
import torch
from scores import iou_score,dice_score

def train_model(num_epochs,model,train_dataloader,val_dataloader,optimizer,scheduler,loss_fn,device):
    # Store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_train_loss = 0
        train_progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in train_progress_bar:
            optimizer.zero_grad()

            image = batch['image'].to(device)
            points = batch['points'].to(device)
            point_labels = batch['point_labels'].to(device)
            boxes = batch['boxes'].to(device)
            ground_truth_mask = batch['ground_truth_mask'].to(device)
            original_image_size = tuple(batch['original_image_size'][0].cpu().numpy())
            points_mask = batch['points_mask'].to(device)
            boxes_mask = batch['boxes_mask'].to(device)

            # Forward pass
            mask_logits = model(
                image=image,
                points=points,
                point_labels=point_labels,
                boxes=boxes,
                original_image_size=original_image_size,
                points_mask=points_mask,
                boxes_mask=boxes_mask
            )

            # For simplicity, assume a single mask prediction for loss calculation.
            # If num_mask_tokens > 1, typically a matching strategy is used to pick the best mask or average them.
            # Here, we'll just take the first predicted mask or ensure it matches the ground truth channel.
            if mask_logits.shape[1] > 1 and ground_truth_mask.shape[1] == 1: # Model outputs multiple masks, GT is single
                predicted_mask_for_loss = mask_logits[:, 0:1, :, :] # Take the first predicted mask
            elif mask_logits.shape[1] == 1 and ground_truth_mask.shape[1] == 1: # Both are single masks
                predicted_mask_for_loss = mask_logits
            else:
                raise ValueError(f"Mask logits shape {mask_logits.shape} not compatible with ground truth mask shape {ground_truth_mask.shape}")

            loss = loss_fn(predicted_mask_for_loss, ground_truth_mask)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        scheduler.step() # Update learning rate

        # Validation loop
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        total_val_iou = 0
        total_val_dice = 0
        val_progress_bar = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")

        with torch.no_grad(): # No gradient calculations during validation
            for batch in val_progress_bar:
                image = batch['image'].to(device)
                points = batch['points'].to(device)
                point_labels = batch['point_labels'].to(device)
                boxes = batch['boxes'].to(device)
                ground_truth_mask = batch['ground_truth_mask'].to(device)
                original_image_size = tuple(batch['original_image_size'][0].cpu().numpy())
                points_mask = batch['points_mask'].to(device)
                boxes_mask = batch['boxes_mask'].to(device)

                mask_logits = model(
                    image=image,
                    points=points,
                    point_labels=point_labels,
                    boxes=boxes,
                    original_image_size=original_image_size,
                    points_mask=points_mask,
                    boxes_mask=boxes_mask
                )

                if mask_logits.shape[1] > 1 and ground_truth_mask.shape[1] == 1:
                    predicted_mask_for_loss = mask_logits[:, 0:1, :, :]
                elif mask_logits.shape[1] == 1 and ground_truth_mask.shape[1] == 1:
                    predicted_mask_for_loss = mask_logits
                else:
                    raise ValueError(f"Mask logits shape {mask_logits.shape} not compatible with ground truth mask shape {ground_truth_mask.shape}")

                loss = loss_fn(predicted_mask_for_loss, ground_truth_mask)
                total_val_loss += loss.item()

                # Calculate metrics
                batch_iou = iou_score(predicted_mask_for_loss, ground_truth_mask)
                batch_dice = dice_score(predicted_mask_for_loss, ground_truth_mask)
                total_val_iou += batch_iou.item()
                total_val_dice += batch_dice.item()
                val_progress_bar.set_postfix(loss=loss.item(), iou=batch_iou.item(), dice=batch_dice.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_iou = total_val_iou / len(val_dataloader)
        avg_val_dice = total_val_dice / len(val_dataloader)

        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        history['val_dice'].append(avg_val_dice)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}")

    print("Training complete.")
    return history
