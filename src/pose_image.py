import cv2
import mediapipe as mp
import json
import sys

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

image = cv2.imread('sample.jpg')
if image is None:
    print("Error: 'sample.jpg' not found or couldn't be loaded.")
    sys.exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Instead of imshow, save the image to avoid GUI errors
    cv2.imwrite("pose_output.png", image)
    print("Pose estimation output saved as pose_output.png")

    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append({'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility})
    with open('keypoints.json', 'w') as f:
        json.dump(keypoints, f, indent=4)
else:
    print("No pose landmarks found in the image.")
