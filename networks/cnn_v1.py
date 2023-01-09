from tensorflow._api.v2.v2.keras import Model
from tensorflow._api.v2.v2.keras.layers import (
    Dense,
    Conv1D,
    BatchNormalization,
    LeakyReLU,
    GlobalAveragePooling1D,
    Input,
    Dropout,
    Layer,
)
from networks.interface_model_utilities import InterfaceModelUtilities


class CNNv1(Model, InterfaceModelUtilities):
    def __init__(self, num_classes: int = 5, sampling_rate: int = 100):
        super().__init__()
        self._sampling_rate = sampling_rate
        self._input_shape = (sampling_rate * 10, 12)

        self.cnn_block_1 = CNNBlock(filters=32)
        self.cnn_block_2 = CNNBlock(filters=64)
        self.cnn_block_3 = CNNBlock(filters=128)
        self.cnn_block_4 = CNNBlock(filters=64)
        self.cnn_block_5 = CNNBlock(filters=32)
        self.global_average_pooling = GlobalAveragePooling1D()
        self.classifier = Dense(units=num_classes, activation="sigmoid")

    def call(self, inputs, training=False, mask=None):
        x = self.cnn_block_1(inputs, training=training)
        x = self.cnn_block_2(x, training=training)
        x = self.cnn_block_3(x, training=training)
        x = self.cnn_block_4(x, training=training)
        x = self.cnn_block_5(x, training=training)
        x = self.global_average_pooling(x)
        return self.classifier(x)

    def model_architecture(self) -> Model:
        x = Input(shape=self._input_shape)
        return Model(inputs=[x], outputs=self.call(x))

    def model_name(self) -> str:
        if self._sampling_rate != 100:
            return f"cnn_v1_{self._sampling_rate}"
        return f"cnn_v1"

    def sampling_rate(self) -> int:
        return self._sampling_rate

    def need_3D_input(self) -> bool:
        return False


class CNNBlock(Layer):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")
        self.batch_norm = BatchNormalization()
        self.leaky_relu = LeakyReLU()
        self.dropout = Dropout(rate=0.2)

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.leaky_relu(x)
        return self.dropout(x)
