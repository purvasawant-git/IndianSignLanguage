import os
import cv2
import mediapipe as mp
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_path = 'dataset_ISL'
classes = os.listdir(dataset_path)

total_checked = 0
total_detected = 0

for c in classes:
    class_path = os.path.join(dataset_path, c)
    if os.path.isdir(class_path):
        files = os.listdir(class_path)
        sample_files = random.sample(files, min(10, len(files)))
        for f in sample_files:
            img_path = os.path.join(class_path, f)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                total_checked += 1
                if results.multi_hand_landmarks:
                    total_detected += 1

print(f"Checked: {total_checked}, Detected: {total_detected}, Detection Rate: {total_detected/total_checked*100:.2f}%")
