from tensorflow._api.v2.v2.keras import Model
from tensorflow._api.v2.v2.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Input,
)
from networks.interface_model_utilities import InterfaceModelUtilities


class CNNv3(Model, InterfaceModelUtilities):
    def __init__(self, num_classes: int = 5, sampling_rate: int = 100):
        super().__init__()
        self._sampling_rate = sampling_rate
        self._input_shape = (sampling_rate * 10, 12, 1)

        self.conv_1 = Conv2D(
            filters=64, kernel_size=7, activation="relu", padding="same"
        )
        self.max_pool_1 = MaxPooling2D(pool_size=2)
        self.conv_2 = Conv2D(
            filters=128, kernel_size=3, activation="relu", padding="same"
        )
        self.conv_3 = Conv2D(
            filters=128, kernel_size=3, activation="relu", padding="same"
        )
        self.max_pool_2 = MaxPooling2D(pool_size=2)
        self.conv_4 = Conv2D(
            filters=256, kernel_size=3, activation="relu", padding="same"
        )
        self.conv_5 = Conv2D(
            filters=256, kernel_size=3, activation="relu", padding="same"
        )
        self.max_pool_3 = MaxPooling2D(pool_size=2)
        self.flatten_1 = Flatten()
        self.dense_1 = Dense(units=128, activation="relu")
        self.dropout_1 = Dropout(rate=0.5)
        self.dense_2 = Dense(units=64, activation="relu")
        self.dropout_2 = Dropout(rate=0.5)
        self.classifier = Dense(num_classes, activation="sigmoid")

    def call(self, inputs, training=False, mask=None):
        x = self.conv_1(inputs, training=training)
        x = self.max_pool_1(x, training=training)
        x = self.conv_2(x, training=training)
        x = self.conv_3(x, training=training)
        x = self.max_pool_2(x, training=training)
        x = self.conv_4(x, training=training)
        x = self.conv_5(x, training=training)
        x = self.max_pool_3(x, training=training)
        x = self.flatten_1(x, training=training)
        x = self.dense_1(x, training=training)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x, training=training)
        x = self.dropout_2(x, training=training)
        return self.classifier(x)

    def model_architecture(self) -> Model:
        x = Input(shape=self._input_shape)
        return Model(inputs=[x], outputs=self.call(x))

    def model_name(self) -> str:
        return f"cnn_v3_{self._sampling_rate}"

    def sampling_rate(self) -> int:
        return self._sampling_rate

    def need_3D_input(self) -> bool:
        return True
