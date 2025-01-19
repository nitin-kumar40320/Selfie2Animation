from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Model

class residual_block(Model):
    def __init__(self, kernel_size, filters):
        super(residual_block, self).__init__()
        self.conv1 = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = 1, padding='same')
        self.norm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = 1, padding='same')
        self.norm2 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return layers.add([x, inputs])

class down_conv(Model):
    def __init__(self, kernel_size, filters):
        super(down_conv, self).__init__()
        self.conv1 = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = 2)
        self.conv2 = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = 1, padding='same')
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.norm(x)
        return self.relu(x)

class up_conv(Model):
    def __init__(self, kernel_size, filters):
        super(up_conv, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = 2)
        self.conv2 = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = 1, padding='same')
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.norm(x)
        return self.relu(x)


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_layer = keras.Sequential([
            layers.Conv2D(filters = 64, kernel_size = 7, strides = 1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.down_conv1 = down_conv(3, 128)
        self.down_conv2 = down_conv(3, 256)

        self.residual_block1 = keras.Sequential([residual_block(3, 256) for i in range(8)])

        self.up_conv1 = up_conv(3, 128)
        self.up_conv2 = up_conv(3, 64)

        self.output_layer = keras.Sequential([layers.Conv2D(filters = 3, kernel_size = 7, strides = 1, padding='same')])

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.down_conv1(x)
        x = self.down_conv2(x)
        x = self.residual_block1(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.output_layer(x)
        x = tf.image.resize_with_crop_or_pad(x, tf.shape(inputs)[1], tf.shape(inputs)[2])
        return x


if __name__ == '__main__':
    model = Generator()