from losses import composite_loss_fn
import tensorflow as tf
import os
import time

# --- Custom Training and Evaluation Steps ---
@tf.function
def train_step(model, optimizer, learning_rate_schedule_obj, batch, global_step, num_recycling_steps):
    current_learning_rate = learning_rate_schedule_obj(global_step)

    mixed_precision_enabled = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'

    with tf.GradientTape() as tape:
        predictions = model(batch, training=True)

        composite_loss, individual_losses = composite_loss_fn(
            predictions=predictions,
            batch=batch,
            num_recycling_steps=num_recycling_steps
        )

        if mixed_precision_enabled and isinstance(optimizer, tf.keras.optimizers.LossScaleOptimizer):
            scaled_loss = optimizer.get_scaled_loss(composite_loss)
        else:
            scaled_loss = composite_loss

    gradients = tape.gradient(scaled_loss, model.trainable_variables)

    if mixed_precision_enabled and isinstance(optimizer, tf.keras.optimizers.LossScaleOptimizer):
        gradients = optimizer.get_unscaled_gradients(gradients)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return composite_loss, individual_losses, current_learning_rate

@tf.function
def eval_step(model, batch, num_recycling_steps):
    predictions = model(batch, training=False)

    composite_loss, individual_losses = composite_loss_fn(
        predictions=predictions,
        batch=batch,
        num_recycling_steps=num_recycling_steps
    )

    return composite_loss, individual_losses

# --- Main Training Loop Orchestration ---
def run_training_loop(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    num_epochs: int,
    num_recycling_steps: int,
    log_interval_steps: int = 100,
    eval_interval_epochs: int = 1,
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    mixed_precision_enabled: bool = False,
    learning_rate_schedule_obj = None
):
    # Setup TensorBoard SummaryWriter
    os.makedirs(os.path.join(log_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'val'), exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

    # Setup Checkpoint Manager
    global_step = tf.Variable(0, dtype=tf.int64, name="global_step")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # Restore latest checkpoint if available
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f'Restored from {ckpt_manager.latest_checkpoint}')
    else:
        print('Initializing from scratch.')

    if mixed_precision_enabled and not isinstance(optimizer, tf.keras.optimizers.LossScaleOptimizer):
        print("Warning: Mixed precision enabled but optimizer is not LossScaleOptimizer." \
              "Ensure your optimizer is wrapped with LossScaleOptimizer for proper gradient scaling.")

    print(f"Starting training for {num_epochs} epochs.")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch in train_dataset:
            global_step.assign_add(1)
            num_train_batches += 1

            train_composite_loss, train_individual_losses, current_lr = train_step(
                model,
                optimizer,
                learning_rate_schedule_obj,
                batch,
                global_step,
                num_recycling_steps
            )
            epoch_train_loss += train_composite_loss

            if global_step % log_interval_steps == 0:
                elapsed_time = time.time() - start_time
                print(f"  Step {global_step.numpy()}, LR: {current_lr.numpy():.6f}, "
                      f"Train Loss: {train_composite_loss.numpy():.4f}, Elapsed: {elapsed_time:.2f}s")
                with train_summary_writer.as_default():
                    tf.summary.scalar('Learning Rate', current_lr, step=global_step)
                    tf.summary.scalar('Total Loss', train_composite_loss, step=global_step)
                    for loss_name, loss_value in train_individual_losses.items():
                        tf.summary.scalar(f'Loss/{loss_name}', loss_value, step=global_step)

        avg_epoch_train_loss = epoch_train_loss / num_train_batches
        print(f"  Epoch {epoch + 1} Average Train Loss: {avg_epoch_train_loss:.4f}")

        if (epoch + 1) % eval_interval_epochs == 0:
            print(f"  Running validation for Epoch {epoch + 1}...")
            epoch_val_loss = 0.0
            num_val_batches = 0
            val_individual_losses_sum = {k: 0.0 for k in train_individual_losses.keys()}

            for batch in val_dataset:
                num_val_batches += 1
                val_composite_loss, val_individual_losses = eval_step(
                    model,
                    batch,
                    num_recycling_steps
                )
                epoch_val_loss += val_composite_loss
                for loss_name, loss_value in val_individual_losses.items():
                    val_individual_losses_sum[loss_name] += loss_value

            avg_epoch_val_loss = epoch_val_loss / num_val_batches
            avg_val_individual_losses = {k: v / num_val_batches for k, v in val_individual_losses_sum.items()}

            print(f"  Epoch {epoch + 1} Average Val Loss: {avg_epoch_val_loss:.4f}")

            with val_summary_writer.as_default():
                tf.summary.scalar('Total Loss', avg_epoch_val_loss, step=global_step)
                for loss_name, loss_value in avg_val_individual_losses.items():
                    tf.summary.scalar(f'Loss/{loss_name}', loss_value, step=global_step)

            ckpt_save_path = ckpt_manager.save()
            print(f"  Saved checkpoint for step {global_step.numpy()} at: {ckpt_save_path}")

    print(f"\nTraining finished after {num_epochs} epochs. Total time: {(time.time() - start_time):.2f}s")
