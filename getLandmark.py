import cv2
import mediapipe as mp
import time
import numpy as np
from torch import FloatTensor as tensor
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=True,
    max_num_faces=1,
)
x_data = []
y_data = []
i = 0
for (root, directories, files) in os.walk("../img_crop/happy"):
    for file in files:
        file_path = os.path.join(root, file)
        img = cv2.imread(file_path)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)
        data = []
        print(file_path, end='')
        try:
            for result in results.multi_face_landmarks:
                for _, lm in enumerate(result.landmark):
                    data.append(lm.x)
                    data.append(lm.y)
            print("  success")
            i += 1
            y_data.append([1])
            x_data.append(data)
        except:
            print("  failure")
for (root, directories, files) in os.walk("../img_crop/none"):
    for file in files:
        file_path = os.path.join(root, file)
        img = cv2.imread(file_path)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)
        data = []
        print(file_path, end='')
        try:
            for result in results.multi_face_landmarks:
                for _, lm in enumerate(result.landmark):
                    data.append(lm.x)
                    data.append(lm.y)
            print("  success")
            i += 1
            y_data.append([0])
            x_data.append(data)
        except:
            print("  failure")
x_data = np.array(x_data)

import csv

with open('x_train.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    for data in x_data:
        wr.writerow(data)
with open('y_train.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    for data in y_data:
        wr.writerow(data)

print(len(x_data))
print(len(y_data))
