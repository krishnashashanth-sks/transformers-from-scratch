from torch.utils.data import  DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataset import TFTPyTorchDataset
from model import TemporalFusionTransformer
from losses import QuantileLoss
from train import train_model
from evaluate import evaluate_model
from inference import predict_model

# Set a random seed for reproducibility
np.random.seed(42)

# Define parameters for the synthetic dataset
num_series = 5
num_timesteps = 100
start_date = pd.to_datetime('2023-01-01')

# Create a list to store data for each series
data = []

for i in range(num_series):
    series_id = f'series_{i+1}'

    # Static covariates
    static_category = np.random.choice(['A', 'B', 'C'])
    static_feature = np.random.rand() * 100

    # Time index for the series
    time_index = pd.date_range(start=start_date, periods=num_timesteps, freq='D')

    # Known future dynamic covariates
    day_of_week = time_index.dayofweek
    month = time_index.month
    is_holiday = np.random.randint(0, 2, num_timesteps) # Synthetic holidays

    # Unknown future dynamic covariates
    temperature = np.random.normal(loc=20, scale=5, size=num_timesteps)

    # Target variable (e.g., sales) with some trend and seasonality
    trend = np.linspace(0, 10, num_timesteps)
    seasonality = 5 * np.sin(np.linspace(0, 4 * np.pi, num_timesteps))
    noise = np.random.normal(loc=0, scale=1, size=num_timesteps)
    sales = (50 + trend + seasonality + 0.5 * temperature + noise).astype(int)
    sales[sales < 0] = 0 # Ensure sales are non-negative

    series_df = pd.DataFrame({
        'date': time_index,
        'series_id': series_id,
        'static_category': static_category,
        'static_feature': static_feature,
        'day_of_week': day_of_week,
        'month': month,
        'is_holiday': is_holiday,
        'temperature': temperature,
        'sales': sales
    })
    data.append(series_df)

# Concatenate all series into a single DataFrame
df = pd.concat(data).reset_index(drop=True)

# Define feature groups
TARGET_VARIABLE = 'sales'
TIME_IDX = 'date'
GROUP_IDS = ['series_id'] # Use series_id as a grouping variable

STATIC_CATEGORICAL_FEATURES = ['static_category']
STATIC_REAL_FEATURES = ['static_feature']

KNOWN_DYNAMIC_CATEGORICAL_FEATURES = ['day_of_week', 'month', 'is_holiday']
KNOWN_DYNAMIC_REAL_FEATURES = [] # No continuous known dynamic in this synthetic example

UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES = []
UNKNOWN_DYNAMIC_REAL_FEATURES = ['temperature'] # Temperature is unknown future dynamic

# Combine all categorical and real features for processing
ALL_CATEGORICAL_FEATURES = STATIC_CATEGORICAL_FEATURES + KNOWN_DYNAMIC_CATEGORICAL_FEATURES + UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES
ALL_REAL_FEATURES = STATIC_REAL_FEATURES + KNOWN_DYNAMIC_REAL_FEATURES + UNKNOWN_DYNAMIC_REAL_FEATURES + [TARGET_VARIABLE]

# Make a copy to avoid modifying the original DataFrame directly
df_processed = df.copy()

# --- Label Encode Categorical Features ---
for col in ALL_CATEGORICAL_FEATURES:
    df_processed[col] = df_processed[col].astype('category')
    df_processed[col + '_encoded'] = df_processed[col].cat.codes

# Update feature lists to use encoded columns
STATIC_CATEGORICAL_FEATURES_ENCODED = [col + '_encoded' for col in STATIC_CATEGORICAL_FEATURES]
KNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED = [col + '_encoded' for col in KNOWN_DYNAMIC_CATEGORICAL_FEATURES]
UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED = [col + '_encoded' for col in UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES]

# --- Scale Real Features ---
scalers = {}
for col in ALL_REAL_FEATURES:
    scaler = MinMaxScaler()
    df_processed[col + '_scaled'] = scaler.fit_transform(df_processed[[col]])
    scalers[col] = scaler

# Update feature lists to use scaled columns
STATIC_REAL_FEATURES_SCALED = [col + '_scaled' for col in STATIC_REAL_FEATURES]
KNOWN_DYNAMIC_REAL_FEATURES_SCALED = [col + '_scaled' for col in KNOWN_DYNAMIC_REAL_FEATURES]
UNKNOWN_DYNAMIC_REAL_FEATURES_SCALED = [col + '_scaled' for col in UNKNOWN_DYNAMIC_REAL_FEATURES]
TARGET_VARIABLE_SCALED = TARGET_VARIABLE + '_scaled'

ENCODER_LENGTH = 20  # Number of historical time steps to consider
DECODER_LENGTH = 10  # Number of future time steps to predict

# --- Get feature sizes for TFT initialization ---
static_cat_sizes = get_categorical_sizes(df_processed, STATIC_CATEGORICAL_FEATURES)
known_dynamic_cat_sizes = get_categorical_sizes(df_processed, KNOWN_DYNAMIC_CATEGORICAL_FEATURES)
unknown_dynamic_cat_sizes = get_categorical_sizes(df_processed, UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES)

static_real_size = len(STATIC_REAL_FEATURES_SCALED)
known_dynamic_real_size = len(KNOWN_DYNAMIC_REAL_FEATURES_SCALED)
unknown_dynamic_real_size = len(UNKNOWN_DYNAMIC_REAL_FEATURES_SCALED)

train_dataset = TFTPyTorchDataset(
    data_df=df_model_ready,
    group_ids=GROUP_IDS,
    time_idx=TIME_IDX,
    target_variable_scaled=TARGET_VARIABLE_SCALED,
    static_categorical_features_encoded=STATIC_CATEGORICAL_FEATURES_ENCODED,
    static_real_features_scaled=STATIC_REAL_FEATURES_SCALED,
    known_dynamic_categorical_features_encoded=KNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED,
    known_dynamic_real_features_scaled=KNOWN_DYNAMIC_REAL_FEATURES_SCALED,
    unknown_dynamic_categorical_features_encoded=UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED,
    unknown_dynamic_real_features_scaled=UNKNOWN_DYNAMIC_REAL_FEATURES_SCALED,
    encoder_length=ENCODER_LENGTH,
    decoder_length=DECODER_LENGTH,
    min_prediction_idx_for_target=(ENCODER_LENGTH + DECODER_LENGTH - 1), # Earliest possible start of pred window
    max_prediction_idx_for_target=train_max_target_idx_exclusive
)

val_dataset = TFTPyTorchDataset(
    data_df=df_model_ready,
    group_ids=GROUP_IDS,
    time_idx=TIME_IDX,
    target_variable_scaled=TARGET_VARIABLE_SCALED,
    static_categorical_features_encoded=STATIC_CATEGORICAL_FEATURES_ENCODED,
    static_real_features_scaled=STATIC_REAL_FEATURES_SCALED,
    known_dynamic_categorical_features_encoded=KNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED,
    known_dynamic_real_features_scaled=KNOWN_DYNAMIC_REAL_FEATURES_SCALED,
    unknown_dynamic_categorical_features_encoded=UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED,
    unknown_dynamic_real_features_scaled=UNKNOWN_DYNAMIC_REAL_FEATURES_SCALED,
    encoder_length=ENCODER_LENGTH,
    decoder_length=DECODER_LENGTH,
    min_prediction_idx_for_target=train_max_target_idx_exclusive,
    max_prediction_idx_for_target=val_max_target_idx_exclusive
)

test_dataset = TFTPyTorchDataset(
    data_df=df_model_ready,
    group_ids=GROUP_IDS,
    time_idx=TIME_IDX,
    target_variable_scaled=TARGET_VARIABLE_SCALED,
    static_categorical_features_encoded=STATIC_CATEGORICAL_FEATURES_ENCODED,
    static_real_features_scaled=STATIC_REAL_FEATURES_SCALED,
    known_dynamic_categorical_features_encoded=KNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED,
    known_dynamic_real_features_scaled=KNOWN_DYNAMIC_REAL_FEATURES_SCALED,
    unknown_dynamic_categorical_features_encoded=UNKNOWN_DYNAMIC_CATEGORICAL_FEATURES_ENCODED,
    unknown_dynamic_real_features_scaled=UNKNOWN_DYNAMIC_REAL_FEATURES_SCALED,
    encoder_length=ENCODER_LENGTH,
    decoder_length=DECODER_LENGTH,
    min_prediction_idx_for_target=val_max_target_idx_exclusive,
    max_prediction_idx_for_target=test_max_target_idx_exclusive
)

# Initialize the DataLoaders
batch_size = 32 # You can adjust this
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of validation batches: {len(val_dataloader)}")
print(f"Number of test batches: {len(test_dataloader)}")

# Model Hyperparameters (Example values, these would be tuned)
hidden_size = 64
num_heads = 4
num_gru_layers = 1
dropout_rate = 0.1
output_quantiles = [0.1, 0.5, 0.9] # Example quantiles
learning_rate = 1e-3

# Instantiate the Quantile Loss
loss_function = QuantileLoss(output_quantiles)

print("QuantileLoss function defined and instantiated.")
print(f"Output Quantiles: {output_quantiles}")

# Instantiate the TFT Model (assuming all feature sizes are defined from previous steps)
model = TemporalFusionTransformer(
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_gru_layers=num_gru_layers,
    dropout_rate=dropout_rate,
    output_quantiles=output_quantiles,
    static_categorical_sizes=static_cat_sizes,
    static_real_size=static_real_size,
    known_dynamic_categorical_sizes=known_dynamic_cat_sizes,
    known_dynamic_real_size=known_dynamic_real_size,
    unknown_dynamic_categorical_sizes=unknown_dynamic_cat_sizes,
    unknown_dynamic_real_size=unknown_dynamic_real_size,
    encoder_length=ENCODER_LENGTH,
    decoder_length=DECODER_LENGTH
)

# Instantiate the Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("TFT Model and Optimizer instantiated.")
print(f"Model parameters count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Ensure device is defined (if not already)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model.to(device)

# Number of epochs (for demonstration, let's just run one epoch)
num_epochs = 1

train_model(num_epochs,model,train_dataloader,optimizer,loss_function,device)

# Evaluate on the validation set
val_loss = evaluate_model(model, val_dataloader, loss_function, device)
print(f"Validation Loss: {val_loss:.4f}")

# Generate predictions on the test set
test_predictions, test_targets = predict_model(model, test_dataloader, device)

print(f"Test Predictions shape: {test_predictions.shape}")
print(f"Test Targets shape: {test_targets.shape}")

# Display a few predictions and corresponding targets (e.g., for the first sample in the batch)
print("\nFirst sample's predictions (quantiles):")
print(test_predictions[0])
print("\nFirst sample's actual targets:")
print(test_targets[0])