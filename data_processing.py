import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
from PIL import Image, ImageFilter

real_images_dir = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\anime_faces"
real_edge_smoothed_images = "D:\\Kaam_ke_projects\\selfie2anime\\final_data\\anime_face_edge_smoothed"

def smoothen_edges(input_image_path, output_path):

    image = Image.open(input_image_path)
    smoothed_image = image.filter(ImageFilter.SMOOTH_MORE)
    smoothed_image.save(output_path)


counter = 0
temp_file_names = os.listdir(real_images_dir)
temp_file_names.sort()

for file_name in temp_file_names:
        source_file = os.path.join(real_images_dir, file_name)
        destination_file = os.path.join(real_edge_smoothed_images, file_name)
        smoothen_edges(source_file, destination_file)
        print(f"Copied: {source_file} -> {destination_file}")
        counter+=1

print(f"Total files copied: {counter}")
