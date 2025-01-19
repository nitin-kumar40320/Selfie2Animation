from tensorflow import keras
from tensorflow.keras import layers, Model

class forward(Model):
    def __init__(self, kernel_size, filters):
        super(forward, self).__init__()
        self.conv1 = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = 2)
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.conv2 = layers.Conv2D(filters = filters * 2, kernel_size = kernel_size, strides = 1, padding='same')
        self.norm = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.norm(x)
        return self.lrelu(x)

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_layer = keras.Sequential([
            layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding='same'),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.forward1 = forward(3, 64)
        self.forward2 = forward(3, 128)
        self.forward3 = keras.Sequential([
            layers.Conv2D(filters = 256, kernel_size = 3, strides = 1),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.output_layer = keras.Sequential([layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding='same')])

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.forward1(x)
        x = self.forward2(x)
        x = self.forward3(x)
        return self.output_layer(x)


if __name__ == "__main__":
    discriminator = Discriminator()