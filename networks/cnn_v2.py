from keras import Model
from keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dropout,
    Layer,
    Input,
)


class Cnn(Model):
    def __init__(self, num_classes=5, input_shape=(1000, 12, 1)):
        super().__init__()
        self._input_shape = input_shape
        self.cnn_block_1 = CNNBlock(32)
        self.cnn_block_2 = CNNBlock(64)
        self.cnn_block_3 = CNNBlock(128)
        self.cnn_block_4 = CNNBlock(256)
        self.cnn_block_5 = CNNBlock(512)
        self.global_average_pooling = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation="sigmoid")

    def call(self, inputs, training=False, mask=None):
        x = self.cnn_block_1(inputs, training=training)
        x = self.cnn_block_2(x, training=training)
        x = self.cnn_block_3(x, training=training)
        x = self.cnn_block_4(x, training=training)
        x = self.cnn_block_5(x, training=training)
        x = self.global_average_pooling(x)
        return self.classifier(x)

    def model(self):
        x = Input(shape=self._input_shape)
        return Model(inputs=[x], outputs=self.call(x))


class CNNBlock(Layer):
    def __init__(self, out_channels, kernel_size=3):
        super().__init__()
        self.conv = Conv2D(
            filters=out_channels, kernel_size=kernel_size, padding="same"
        )
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        self.dropout = Dropout(rate=0.2)

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        return self.dropout(x)
