from tensorflow import keras
VOCAB_SIZE=1000
SEQUENCE_LENGTH=512
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE, skip_top=0, oov_token=2
)

x_train_padded = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=SEQUENCE_LENGTH)
x_test_padded = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=SEQUENCE_LENGTH)
