from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Input,
    Bidirectional,
    LSTM,
    Dropout,
)


class Lstm(keras.Model):
    def __init__(self):
        super().__init__()
        self.LSTM_1 = Bidirectional(LSTM(units=32, return_sequences=True))
        self.dropout_1 = Dropout(rate=0.1)
        self.LSTM_2 = Bidirectional(LSTM(units=16))
        self.dropout_2 = Dropout(rate=0.1)
        self.dense_1 = Dense(64, activation="relu")
        self.dense_2 = Dense(32, activation="relu")
        self.dense_3 = Dense(16, activation="relu")
        self.dense_4 = Dense(5, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.LSTM_1(inputs)
        if training:
            x = self.dropout_1(x)
        x = self.LSTM_2(x)
        if training:
            x = self.dropout_2(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.dense4(x)
