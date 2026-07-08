import tensorflow as tf
import numpy as np
from utils import create_informer_data_sample

#  Custom Keras Sequence for generating batches
class InformerDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, time_features, seq_len, label_len, pred_len, batch_size, shuffle=True):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)

        self.max_start_idx = self.n_samples - (self.seq_len + self.pred_len)

        if self.max_start_idx < 0:
            raise ValueError("Not enough data to create any sample with the given sequence lengths.")

        self.indices = np.arange(self.max_start_idx + 1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_enc_input = []
        batch_dec_input = []
        batch_enc_time_features = []
        batch_dec_time_features = []
        batch_target_output = []

        for start_idx in batch_indices:
            enc_input, dec_input, enc_time_feat, dec_time_feat, target_output = \
                create_informer_data_sample(self.data, self.time_features, start_idx, self.seq_len, self.label_len, self.pred_len)

            batch_enc_input.append(enc_input)
            batch_dec_input.append(dec_input)
            batch_enc_time_features.append(enc_time_feat)
            batch_dec_time_features.append(dec_time_feat)
            batch_target_output.append(target_output)

        return (
            [np.array(batch_enc_input),
             np.array(batch_dec_input),
             np.array(batch_enc_time_features),
             np.array(batch_dec_time_features)],
            np.array(batch_target_output)
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
