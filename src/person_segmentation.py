import cv2
from rembg import remove
import os

input_folder = "data/person_images"      # Folder containing person images
output_folder = "data/person_masked"     # Folder to save masked images

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_masked.png')

        input_image = cv2.imread(input_path)
        if input_image is None:
            print(f"Error reading {input_path}")
            continue

        output_image = remove(input_image)
        cv2.imwrite(output_path, output_image)
        print(f"Processed {input_path} -> {output_path}")

print("Person segmentation batch complete.")
