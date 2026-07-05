import tensorflow as tf
import numpy as np
import os
import tempfile
import shutil
from utils import create_dummy_data,create_alphafold_dataset
from model import AlphaFoldModel
from decay import WarmupCosineDecay
from train import run_training_loop

# --- 1. Data Preparation: Create Dummy Data and Datasets ---
# These functions are assumed to be defined in previous cells
# (create_dummy_data, create_alphafold_dataset)

# Configuration for dummy data
NUM_TRAIN_PROTEINS = 5
NUM_VAL_PROTEINS = 2
BATCH_SIZE = 1 # Batch size for training/validation

# Create temporary directories for dummy data
base_tmp_dir_train = tempfile.mkdtemp()
base_tmp_dir_val = tempfile.mkdtemp()

dummy_data_paths_train = {
    'a3m_dir': os.path.join(base_tmp_dir_train, 'a3m_dir'),
    'pdb_dir': os.path.join(base_tmp_dir_train, 'pdb_dir'),
    'template_info_dir': os.path.join(base_tmp_dir_train, 'template_info_dir')
}
dummy_data_paths_val = {
    'a3m_dir': os.path.join(base_tmp_dir_val, 'a3m_dir'),
    'pdb_dir': os.path.join(base_tmp_dir_val, 'pdb_dir'),
    'template_info_dir': os.path.join(base_tmp_dir_val, 'template_info_dir')
}

train_protein_ids = [f"train_protein_{i}" for i in range(NUM_TRAIN_PROTEINS)]
val_protein_ids = [f"val_protein_{i}" for i in range(NUM_VAL_PROTEINS)]

print("Generating dummy training data...")
for protein_id in train_protein_ids:
    create_dummy_data(base_tmp_dir_train, protein_id,
                      sequence_length=np.random.randint(15, 25), # Random length between 15-24
                      num_msa_sequences=np.random.randint(3, 7)) # Random MSA depth between 3-6

print("Generating dummy validation data...")
for protein_id in val_protein_ids:
    create_dummy_data(base_tmp_dir_val, protein_id,
                      sequence_length=np.random.randint(15, 25),
                      num_msa_sequences=np.random.randint(3, 7))

print("Creating tf.data.Dataset for training...")
train_dataset = create_alphafold_dataset(
    protein_ids=train_protein_ids,
    data_paths=dummy_data_paths_train,
    batch_size=BATCH_SIZE,
    shuffle_buffer_size=NUM_TRAIN_PROTEINS
)

print("Creating tf.data.Dataset for validation...")
val_dataset = create_alphafold_dataset(
    protein_ids=val_protein_ids,
    data_paths=dummy_data_paths_val,
    batch_size=BATCH_SIZE,
    shuffle_buffer_size=NUM_VAL_PROTEINS
)

print("Data preparation complete.")

# --- 2. Model Instantiation ---
# The AlphaFoldModel class is assumed to be defined in a previous cell.

# Model Hyperparameters (simplified for demonstration)
c_msa_model = 256  # MSA feature dimension
c_pair_model = 128 # Pairwise feature dimension
NUM_RECYCLING_STEPS_MODEL = 1 # Number of recycling steps

print("Instantiating AlphaFoldModel...")
model = AlphaFoldModel(
    c_msa=c_msa_model,
    c_pair=c_pair_model,
    num_recycling_steps=NUM_RECYCLING_STEPS_MODEL,
    # Default IPA and prediction head params are used if not specified
)

# --- 3. Optimizer and Learning Rate Schedule Instantiation ---
# WarmupCosineDecay and AdamW are assumed to be defined in previous cells.

INITIAL_LR_OPT = 1e-3
TOTAL_TRAINING_STEPS_OPT = 10 * (NUM_TRAIN_PROTEINS // BATCH_SIZE) # Example total steps
WARMUP_STEPS_OPT = 1 # Example warmup for 1 step

print("Creating learning rate schedule...")
learning_rate_schedule = WarmupCosineDecay(
    initial_learning_rate=INITIAL_LR_OPT,
    decay_steps=TOTAL_TRAINING_STEPS_OPT - WARMUP_STEPS_OPT,
    warmup_steps=WARMUP_STEPS_OPT
)

print("Creating AdamW optimizer...")
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=learning_rate_schedule,
    weight_decay=1e-4, # Example weight decay value
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6
)

# --- 4. Training Loop Execution ---
# The run_training_loop function is assumed to be defined in a previous cell.

CHECKPOINT_DIR = './af_dummy_checkpoints'
LOG_DIR = './af_dummy_logs'
NUM_EPOCHS = 2 # Small number of epochs for quick demonstration

# Clean up previous runs if they exist
if os.path.exists(CHECKPOINT_DIR): shutil.rmtree(CHECKPOINT_DIR)
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'val'), exist_ok=True)

print("Starting training loop...")
run_training_loop(
    model=model,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=NUM_EPOCHS,
    num_recycling_steps=NUM_RECYCLING_STEPS_MODEL,
    log_interval_steps=1, # Log every step for dummy run
    eval_interval_epochs=1, # Evaluate every epoch
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    mixed_precision_enabled=False, # Set to True if using mixed precision
    learning_rate_schedule_obj=learning_rate_schedule
)

print("Training loop complete.")

# --- 5. Basic Inference Example (after training) ---
print("\nDemonstrating a basic inference step...")

# Get one batch from the validation dataset for inference
for inference_batch in val_dataset.take(1):
    # The input features are nested under 'inputs' in the batch dict when passed to model.
    # For direct call, we pass the dict as expected by model.call()
    
    # Ensure the batch dictionary has the expected keys for model.call
    # The create_alphafold_dataset function returns a dict that is already structured for model.call
    
    # Extract input features from the batch
    msa_input_features = inference_batch['msa_one_hot']
    pair_input_features = inference_batch['co_evolutionary_features']
    template_features = {k: v for k, v in inference_batch.items() if k.startswith('template_')}
    msa_mask = inference_batch['msa_mask']
    pair_mask = inference_batch['pair_mask']
    atom_mask = inference_batch['atom_mask']

    inference_inputs = {
        "msa_one_hot": msa_input_features,
        "pair_input_features": pair_input_features, # Renamed to match model.call expectation
        "template_features": template_features, # Pass template_features as a dict
        "msa_mask": msa_mask,
        "pair_mask": pair_mask,
        "atom_mask": atom_mask
    }

    # Perform inference
    predictions = model(inference_inputs, training=False)

    print("Inference predictions (shapes):")
    for key, value in predictions.items():
        if isinstance(value, list):
            print(f"  {key}: [{', '.join([str(v.shape) for v in value])}]")
        elif hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    break # Process only one batch for demonstration

print("Inference demonstration complete.")

# --- 6. Cleanup ---
print("\nCleaning up dummy data and checkpoint directories...")
shutil.rmtree(base_tmp_dir_train)
shutil.rmtree(base_tmp_dir_val)
shutil.rmtree(CHECKPOINT_DIR)
shutil.rmtree(LOG_DIR)
print("Cleanup complete.")

print("Full AlphaFold v2 pipeline demonstration finished.")