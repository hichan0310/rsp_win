import torch
import cv2
import mediapipe as mp
import numpy as np
from torch import FloatTensor as tensor
import torch.nn as nn
import matplotlib.pyplot as plt

class learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(956, 50),
            nn.ReLU(),
            nn.Linear(50, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


model = learning()
model.load_state_dict(torch.load('regression_model.pth'))
model.eval()

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=True,
    max_num_faces=1,
)

print('Running')
error = "No Face"

history=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
while True:
    res = ""
    success, img = cap.read()
    if not success:
        break
    results = face_mesh.process(img)  # 여기에 추출한 손 정보가 저장됨
    if not (type(results.multi_face_landmarks) is list):
        error = "No Face"
        cropped = img
        cv2.putText(cropped, error, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        error = ""
        cv2.imshow("img", img)
        cv2.imshow("debug", cropped)
        cv2.waitKey(1)
        continue

    x_max = 0
    y_max = 0
    x_min = 1
    y_min = 1

    for result in results.multi_face_landmarks:
        for _, lm in enumerate(result.landmark):
            if lm.x > x_max: x_max = lm.x
            if lm.y > y_max: y_max = lm.y
            if lm.x < x_min: x_min = lm.x
            if lm.y < y_min: y_min = lm.y

    dx = 480 * (x_max - x_min)
    dy = 640 * (y_max - y_min)

    cropped = img[int(max(480 * y_min - dx / 3 - dx / 4, 0)):int(min(480 * y_max + dx / 3, 479)),
              int(max(640 * x_min - dy / 3, 0)):int(min(640 * x_max + dy / 3, 639))]
    cv2.imwrite("now_img.jpg", cropped)

    try:
        data = []
        results = face_mesh.process(cropped)
        for result in results.multi_face_landmarks:
            for _, lm in enumerate(result.landmark):
                data.append(lm.x)
                data.append(lm.y)
        data = np.array(data)
        data = data.reshape(-1)

        prediction = model(tensor(data).view(1, -1))
        prediction = prediction.detach().numpy()
        res = prediction[0]
    except:
        error = "FaceMesh Error"
        res=0

    cropped = cv2.imread("now_img.jpg")
    cv2.putText(cropped, error, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    error = ""
    cv2.imshow("img", img)
    cv2.imshow("debug", cropped)
    history.append(res)
    history=history[1:20]
    plt.ylim([0, 1])
    plt.plot(range(19), history)
    print('\r', float(res), end='')
    plt.savefig('savefig.png')
    plt.clf()
    cv2.imshow("plot", cv2.imread('savefig.png'))
    cv2.waitKey(1)
