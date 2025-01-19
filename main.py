import numpy as np
import os

from loss_functions import  VGGPerceptualLoss, Discriminator_loss, Generator_loss
from Generator_model import Generator
from Discriminator_model import Discriminator
from dataloader import DataLoader
from trainer import GANTrainer, GeneratorTrainer

print("the code starts")

input_dir = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\human_face"
real_dir = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\anime_faces"
real_dir_edge_smoothed = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\anime_face_edge_smoothed"
batch_size = 8

data_loader = DataLoader(input_dir, real_dir, real_dir_edge_smoothed, batch_size)

generator_model = Generator()
discriminator_model = Discriminator()

generator_initializer = GeneratorTrainer(generator_model, learning_rate=1e-4)
trainer = GANTrainer(generator_model, discriminator_model, content_loss_weight = 10, learning_rate=1e-4)

print("Initialisations are done")

dataset = data_loader.dataset

print("Dataset is fetched")

generator_initializer.train(dataset, epochs=1)

print("Generator is initialised")