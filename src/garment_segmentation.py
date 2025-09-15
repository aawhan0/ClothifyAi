import cv2
from rembg import remove
from PIL import Image
import numpy as np

input_path = 'garment.jpg'         # Your garment image path
output_path = 'garment_masked.png' # Output path with background removed

input_image = cv2.imread(input_path)
if input_image is None:
    print(f"Error: {input_path} not found!")
    exit()

# Convert to RGB for rembg & PIL compatibility
input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(input_rgb)

# Remove background
output_pil = remove(pil_img)

# Convert back to OpenCV format (BGRA with alpha)
output_np = np.array(output_pil)
output_image = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGRA)

# Save the result with alpha channel
cv2.imwrite(output_path, output_image)
print(f"Garment background removed. Output saved as {output_path}")
