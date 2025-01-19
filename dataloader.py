import tensorflow as tf
import os

class DataLoader:
    def __init__(self, input_dir, real_dir, real_dir_edge_smoothed, batch_size=32):
        self.input_dir = input_dir
        self.real_dir = real_dir
        self.real_dir_edge_smoothed = real_dir_edge_smoothed
        self.batch_size = batch_size
        self.dataset = self._create_dataset()

    def _load_and_preprocess_image(self, input_path, real_path, real_path_edge_smoothed):
        input_image = tf.io.read_file(input_path)
        input_image = tf.image.decode_image(input_image, channels=3)
        input_image = tf.cast(input_image, tf.float32) / 255.0

        real_image = tf.io.read_file(real_path)
        real_image = tf.image.decode_image(real_image, channels=3)
        real_image = tf.cast(real_image, tf.float32) / 255.0

        real_image_edge_smoothed = tf.io.read_file(real_path_edge_smoothed)
        real_image_edge_smoothed = tf.image.decode_image(real_image_edge_smoothed, channels=3)
        real_image_edge_smoothed = tf.cast(real_image_edge_smoothed, tf.float32) / 255.0

        return input_image, real_image, real_image_edge_smoothed

    def _create_dataset(self):
        input_paths = sorted([os.path.join(self.input_dir, fname) for fname in os.listdir(self.input_dir)])
        real_paths = sorted([os.path.join(self.real_dir, fname) for fname in os.listdir(self.real_dir)])
        real_paths_edge_smoothed = sorted([os.path.join(self.real_dir_edge_smoothed, fname) for fname in os.listdir(self.real_dir_edge_smoothed)])

        if len(input_paths) != len(real_paths) or len(input_paths) != len(real_paths_edge_smoothed):
            raise ValueError("The number of input and real and real edge_smoothed images must be the same.")

        dataset = tf.data.Dataset.from_tensor_slices((input_paths, real_paths, real_paths_edge_smoothed))
        dataset = dataset.map(self._load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def get_dataset(self):
        return self.dataset


if __name__ == "__main__":
    input_dir = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\human_face"
    real_dir = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\anime_faces"
    real_dir_edge_smoothed = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\anime_face_edge_smoothed"
    batch_size = 32

    data_loader = DataLoader(input_dir, real_dir, real_dir_edge_smoothed, batch_size)