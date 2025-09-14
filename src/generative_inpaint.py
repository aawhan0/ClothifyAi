import cv2

img = cv2.imread('tryon_result.png')
mask = cv2.imread('mask_for_inpainting.png', 0)  # Load a black-and-white mask

inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
cv2.imwrite('tryon_final.png', inpainted_img)
print("Inpainting done. Output: tryon_final.png")
