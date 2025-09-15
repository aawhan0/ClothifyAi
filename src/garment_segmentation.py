import cv2
from rembg import remove
from PIL import Image
import numpy as np
import os

input_folder = "data/garment_images"     # Folder containing garment images
output_folder = "data/garment_masked"    # Folder to save masked images

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_masked.png')

        input_image = cv2.imread(input_path)
        if input_image is None:
            print(f"Error reading {input_path}")
            continue

        input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(input_rgb)
        output_pil = remove(pil_img)

        output_np = np.array(output_pil)
        output_image = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGRA)

        cv2.imwrite(output_path, output_image)
        print(f"Processed {input_path} -> {output_path}")

print("Garment segmentation batch complete.")
