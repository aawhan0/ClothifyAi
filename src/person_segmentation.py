import cv2
from rembg import remove

#declaring paths of input and output:
input_path = "sample.jpg" #image of the person
output_path = "person_masked.png"

#feeding in functions on input and output
input_image = cv2.imread(input_path)
output_image = remove(input_image)
cv2.imwrite(output_path, output_image)
print("Person segmentation done. Output: ", output_path)
