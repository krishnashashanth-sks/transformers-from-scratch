import torch
from torch.utils.data import Dataset
import numpy as np

class TFTPyTorchDataset(Dataset):
    def __init__(
        self,
        data_df,
        group_ids,
        time_idx,
        target_variable_scaled,
        static_categorical_features_encoded,
        static_real_features_scaled,
        known_dynamic_categorical_features_encoded,
        known_dynamic_real_features_scaled,
        unknown_dynamic_categorical_features_encoded,
        unknown_dynamic_real_features_scaled,
        encoder_length,
        decoder_length,
        min_prediction_idx_for_target: int, # Minimum index of the last target of a sample
        max_prediction_idx_for_target: int  # Maximum index (exclusive) of the last target of a sample
    ):
        self.data_df = data_df.copy()
        self.group_ids = group_ids
        self.time_idx = time_idx
        self.target_variable_scaled = target_variable_scaled
        self.static_categorical_features_encoded = static_categorical_features_encoded
        self.static_real_features_scaled = static_real_features_scaled
        self.known_dynamic_categorical_features_encoded = known_dynamic_categorical_features_encoded
        self.known_dynamic_real_features_scaled = known_dynamic_real_features_scaled
        self.unknown_dynamic_categorical_features_encoded = unknown_dynamic_categorical_features_encoded
        self.unknown_dynamic_real_features_scaled = unknown_dynamic_real_features_scaled
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.min_prediction_idx_for_target = min_prediction_idx_for_target
        self.max_prediction_idx_for_target = max_prediction_idx_for_target

        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for group_id in self.data_df[self.group_ids[0]].unique():
            series_df = self.data_df[self.data_df[self.group_ids[0]] == group_id].sort_values(by=self.time_idx).reset_index(drop=True)
            series_len = len(series_df)

            # Calculate `i`'s range more directly:
            # `i` is the start of the encoder window.
            # The last index of the prediction window is `i + self.encoder_length + self.decoder_length - 1`

            # Minimum start index `i` such that `end_of_prediction_window_idx >= self.min_prediction_idx_for_target`
            # This translates to: `i >= self.min_prediction_idx_for_target - (self.encoder_length + self.decoder_length - 1)`
            min_i_for_series_absolute = self.min_prediction_idx_for_target - (self.encoder_length + self.decoder_length - 1)
            min_valid_i_for_series = max(0, min_i_for_series_absolute)

            # Maximum start index `i` such that `end_of_prediction_window_idx < self.max_prediction_idx_for_target`
            # This translates to: `i < self.max_prediction_idx_for_target - (self.encoder_length + self.decoder_length - 1)`
            max_i_for_series_absolute = self.max_prediction_idx_for_target - (self.encoder_length + self.decoder_length - 1)
            max_valid_i_for_series = min(series_len - (self.encoder_length + self.decoder_length), max_i_for_series_absolute - 1)

            for i in range(min_valid_i_for_series, max_valid_i_for_series + 1):
                # Static features are constant for the entire series
                static_cat_data = series_df[self.static_categorical_features_encoded].iloc[0].values
                static_real_data = series_df[self.static_real_features_scaled].iloc[0].values

                # Historical (encoder) part
                hist_start = i
                hist_end = i + self.encoder_length
                historical_df = series_df.iloc[hist_start:hist_end]

                hist_known_cat = [historical_df[col].values for col in self.known_dynamic_categorical_features_encoded]
                hist_known_real = historical_df[self.known_dynamic_real_features_scaled].values
                hist_unknown_cat = [historical_df[col].values for col in self.unknown_dynamic_categorical_features_encoded]
                # For unknown real features, target is also considered unknown dynamic for historical part
                hist_unknown_real_cols = self.unknown_dynamic_real_features_scaled + [self.target_variable_scaled]
                # Handle case where hist_unknown_real_cols might be empty if no unknown real features except target
                if not hist_unknown_real_cols:
                    hist_unknown_real = np.empty((self.encoder_length, 0))
                else:
                    hist_unknown_real = historical_df[hist_unknown_real_cols].values

                # Future (decoder) part
                future_start = hist_end
                future_end = hist_end + self.decoder_length
                future_df = series_df.iloc[future_start:future_end]

                future_known_cat = [future_df[col].values for col in self.known_dynamic_categorical_features_encoded]
                future_known_real = future_df[self.known_dynamic_real_features_scaled].values
                future_target = future_df[self.target_variable_scaled].values

                self.samples.append({
                    'static_cat': static_cat_data,
                    'static_real': static_real_data,
                    'historical_known_cat': hist_known_cat,
                    'historical_known_real': hist_known_real,
                    'historical_unknown_cat': hist_unknown_cat,
                    'historical_unknown_real': hist_unknown_real,
                    'future_known_cat': future_known_cat,
                    'future_known_real': future_known_real,
                    'future_target': future_target
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # This collate_fn processes a batch of samples from the TFTPyTorchDataset.
    # It stacks individual tensors and converts lists of tensors to actual PyTorch tensors.

    # Initialize lists to hold collated data for each feature type
    static_cat_batch = []
    static_real_batch = []
    historical_known_cat_batch = []
    historical_known_real_batch = []
    historical_unknown_cat_batch = []
    historical_unknown_real_batch = []
    future_known_cat_batch = []
    future_known_real_batch = []
    future_target_batch = []

    for sample in batch:
        # Static features are (num_static_cat,) and (num_static_real,)
        # Need to handle potential empty lists for categorical or real features
        if sample['static_cat'].size > 0:
            static_cat_batch.append(sample['static_cat'])
        if sample['static_real'].size > 0:
            static_real_batch.append(sample['static_real'])

        # Dynamic features (lists of arrays/tensors)
        historical_known_cat_batch.append(sample['historical_known_cat'])
        historical_known_real_batch.append(sample['historical_known_real'])
        historical_unknown_cat_batch.append(sample['historical_unknown_cat'])
        historical_unknown_real_batch.append(sample['historical_unknown_real'])
        future_known_cat_batch.append(sample['future_known_cat'])
        future_known_real_batch.append(sample['future_known_real'])
        future_target_batch.append(sample['future_target'])

    # Convert static features to tensors
    # Fix: Correctly stack static categorical features per feature, not per sample
    static_cat_output = []
    if static_cat_batch and static_cat_batch[0].size > 0: # Check if there are static categorical features
        num_static_cat_features = static_cat_batch[0].size
        for feature_idx in range(num_static_cat_features):
            feature_values = [arr[feature_idx] for arr in static_cat_batch]
            static_cat_output.append(torch.tensor(feature_values, dtype=torch.long))

    if static_real_batch:
        static_real_output = torch.from_numpy(np.stack(static_real_batch)).float() # (batch_size, num_static_real)
    else:
        static_real_output = torch.empty(len(batch), 0).float()

    # Convert dynamic features to tensors
    # For categorical, it's a list of lists -> need to transpose and stack
    # Example: [[cat1_s1, cat2_s1], [cat1_s2, cat2_s2]] -> [[cat1_s1, cat1_s2], [cat2_s1, cat2_s2]]
    def _to_tensor_list(list_of_lists, dtype=torch.long):
        if not list_of_lists or not list_of_lists[0]: # Handle empty lists or empty inner lists
            return []
        # Transpose and stack, then convert each stacked array to a tensor
        transposed = list(zip(*list_of_lists))
        return [torch.from_numpy(np.stack(l_item)).to(dtype) for l_item in transposed]

    historical_known_cat_output = _to_tensor_list(historical_known_cat_batch, torch.long)
    historical_known_real_output = torch.from_numpy(np.stack(historical_known_real_batch)).float()
    historical_unknown_cat_output = _to_tensor_list(historical_unknown_cat_batch, torch.long)

    if historical_unknown_real_batch and historical_unknown_real_batch[0].shape[-1] > 0: # Check if there are real features to stack
        historical_unknown_real_output = torch.from_numpy(np.stack(historical_unknown_real_batch)).float()
    else:
        historical_unknown_real_output = torch.empty(len(batch), historical_known_real_output.shape[1], 0).float() # (batch_size, encoder_length, 0)

    future_known_cat_output = _to_tensor_list(future_known_cat_batch, torch.long)
    future_known_real_output = torch.from_numpy(np.stack(future_known_real_batch)).float()
    future_target_output = torch.from_numpy(np.stack(future_target_batch)).float()

    return {
        'static_categorical_data': static_cat_output,
        'static_real_data': static_real_output,
        'historical_known_categorical_data': historical_known_cat_output,
        'historical_known_real_data': historical_known_real_output,
        'historical_unknown_categorical_data': historical_unknown_cat_output,
        'historical_unknown_real_data': historical_unknown_real_output,
        'future_known_categorical_data': future_known_cat_output,
        'future_known_real_data': future_known_real_output,
        'future_target': future_target_output
    }
