from tensorflow._api.v2.v2.keras import Model
from tensorflow._api.v2.v2.keras.layers import (
    Dense,
    Bidirectional,
    LSTM,
    Dropout,
    Input,
)
from networks.interface_model_utilities import InterfaceModelUtilities


class LSTM_v1(Model, InterfaceModelUtilities):
    def __init__(self, num_classes: int = 5, sampling_rate: int = 100):
        super().__init__()
        self._sampling_rate = sampling_rate
        self._input_shape = (sampling_rate * 10, 12)

        self.LSTM_1 = Bidirectional(LSTM(units=32, return_sequences=True))
        self.dropout_1 = Dropout(rate=0.1)
        self.LSTM_2 = Bidirectional(LSTM(units=16))
        self.dropout_2 = Dropout(rate=0.1)
        self.dense_1 = Dense(64, activation="relu")
        self.dense_2 = Dense(32, activation="relu")
        self.dense_3 = Dense(16, activation="relu")
        self.dense_4 = Dense(5, activation="sigmoid")

    def call(self, inputs, training=False, mask=None):
        x = self.LSTM_1(inputs, training=training)
        x = self.dropout_1(x, training=training)
        x = self.LSTM_2(x, training=training)
        x = self.dropout_2(x, training=training)
        x = self.dense_1(x, training=training)
        x = self.dense_2(x, training=training)
        x = self.dense_3(x, training=training)
        return self.dense_4(x, training=training)

    def model_architecture(self) -> Model:
        x = Input(shape=self._input_shape)
        return Model(inputs=[x], outputs=self.call(x))

    def model_name(self) -> str:
        return f"lstm_v1_{self._sampling_rate}"

    def sampling_rate(self) -> int:
        return self._sampling_rate

    def need_3D_input(self) -> bool:
        return False
