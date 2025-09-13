import cv2
import mediapipe as mp
import json

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def live_pose_estimation():
    pose = mp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence= 0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab the frame.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Estimation Live', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_pose_estimation()