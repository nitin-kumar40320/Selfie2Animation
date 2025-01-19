import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

class VGGPerceptualLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = VGG19(weights='imagenet', include_top=False)
        self.feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv4').output)
        self.feature_extractor.trainable = False

    def call(self, x_input, y_pred):
        x_input = preprocess_input(x_input * 255.0)
        y_pred = preprocess_input(y_pred * 255.0)
        true_features = self.feature_extractor(x_input)
        pred_features = self.feature_extractor(y_pred)
        loss = tf.reduce_mean(tf.abs(true_features - pred_features))
        return loss


class Discriminator_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(Discriminator_loss, self).__init__()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, real_output, real_output_edge_smoothed, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        real_loss_edge_smoothed = self.cross_entropy(tf.ones_like(real_output_edge_smoothed), real_output_edge_smoothed)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + real_loss_edge_smoothed + fake_loss
        return total_loss


class Generator_loss(tf.keras.losses.Loss):
    def __init__(self, content_loss_weight = 10):
        super(Generator_loss, self).__init__()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.content_loss_function = VGGPerceptualLoss()
        self.content_loss_weight = content_loss_weight

    def call(self, input, fake_image, fake_output):
        cross_entropy_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        content_loss = content_loss_function(input, fake_image)
        return cross_entropy_loss + self.content_loss_weight * content_loss


if __name__ == '__main__':
    loss = VGGPerceptualLoss()
    print(loss)