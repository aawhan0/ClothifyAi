import cv2
import numpy as np
import json

garment_img = cv2.imread('garment.png')
person_img = cv2.imread('person_masked.png')

with open('keypoints.json') as f:
    keypoints = json.load(f)

left_shoulder = (int(keypoints[11]['x'] * person_img.shape[1]), int(keypoints[11]['y'] * person_img.shape[0]))
right_shoulder = (int(keypoints[12]['x'] * person_img.shape[1]), int(keypoints[12]['y'] * person_img.shape[0]))
left_hip = (int(keypoints[23]['x'] * person_img.shape[1]), int(keypoints[23]['y'] * person_img.shape[0]))
right_hip = (int(keypoints[24]['x'] * person_img.shape[1]), int(keypoints[24]['y'] * person_img.shape[0]))

dst_pts = np.float32([left_shoulder, right_shoulder, right_hip, left_hip])
src_h, src_w = garment_img.shape[:2]
src_pts = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped_garment = cv2.warpPerspective(garment_img, M, (person_img.shape[1], person_img.shape[0]))

result = cv2.addWeighted(person_img, 0.7, warped_garment, 0.6, 0)
cv2.imwrite('tryon_result.png', result)
print("Garment warped and composited. Output: tryon_result.png")