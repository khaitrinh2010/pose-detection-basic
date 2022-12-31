import cv2
import pandas as pd
import mediapipe as mp
import tensorflow as tf

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)
# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# List lưu tọa độ các điểm khung xương
point_list = []
temporary_list = []


def land_mark_process(target):
    c_lm = []
    target = target.pose_landmarks.landmark
    for id, lm in enumerate(target):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
    return img



while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # xử lý frameRGB
    results = pose.process(frameRGB)
    # Xuất ra các giá trị về tọa độ của các điểm khung xương và độ visibility
    landmark_process = land_mark_process(results)
    final = draw(mpDraw, results, frame)
    if ret:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
