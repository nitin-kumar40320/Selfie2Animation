from tensorflow.keras.applications import VGG19
import tensorflow as tf

vgg = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.summary() 