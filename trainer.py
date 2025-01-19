import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from loss_functions import VGGPerceptualLoss, Discriminator_loss, Generator_loss

class GANTrainer:
    def __init__(self, generator, discriminator, content_loss_weight = 10, learning_rate=1e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.content_loss_weight = content_loss_weight
        self.discriminator_loss = Discriminator_loss()
        self.generator_loss = Generator_loss(self.content_loss_weight)
        self.generator_optimizer = Adam(learning_rate)
        self.discriminator_optimizer = Adam(learning_rate)

    @tf.function
    def train_step(self, input_image, real_image, real_image_edge_smoothed):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(input_image, training=True)

            real_output = self.discriminator(real_images, training=True)
            real_output_edge_smoothed = self.discriminator(real_image_edge_smoothed, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(input, generated_images)
            disc_loss = self.discriminator_loss(real_output, real_output_edge_smoothed, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for input_image, real_image, real_image_edge_smoothed in dataset:
                gen_loss, disc_loss = self.train_step(input_image, real_image, real_image_edge_smoothed)

            print(f'Epoch {epoch+1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')


class GeneratorTrainer:
    def __init__(self, generator, learning_rate=1e-4):
        self.generator = generator
        self.content_loss = VGGPerceptualLoss()
        self.generator_optimizer = Adam(learning_rate)

    @tf.function
    def train_step(self, input):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(input, training=True)

            gen_loss = self.content_loss(input, generated_images)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss

    # @tf.function
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            print(f"The current epoch is {epoch+1} of {epochs}")
            for i, (input_image, real_image, real_image_edge_smoothed) in enumerate(dataset):
                print(f"Current Batch is {i+1} of {len(dataset)}")
                gen_loss= self.train_step(input_image)

            print(f'Epoch {epoch+1}, Generator Loss: {gen_loss}')
