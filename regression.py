import torch
import cv2
import mediapipe as mp
import time
import numpy as np
from torch import FloatTensor as tensor
import torch.nn as nn


class learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(40, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 4)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer(x)
        return x

model = learning()
model.load_state_dict(torch.load('regression_model.pth'))
model.eval()

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ptime = 0
ctime = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # 여기에 추출한 손 정보가 저장됨

    info = np.zeros((20, 2))

    # 이 부분은 나도 정확히 어떤 과정인지 모름
    # 근데 이렇게 하면 info에 저장이 잘 되더라고
    O_x, O_y = 0, 0
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if id == 0:
                    O_x, O_y = cx, cy
                else:
                    info[id - 1][0], info[id - 1][1] = cx - O_x, cy - O_y
            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)
    # info : np.array, shape=(20, 2), 손바닥 아래 부분을 기준으로 다른 부위들의 상대적인 위치가 저장되어 있음(1~20의 위치벡터)

    prediction = model(tensor(info.reshape((-1))))
    prediction = prediction.detach().numpy()
    print(prediction)

    result = "?"
    ind=-1
    if prediction[1] < prediction[0] > prediction[2] and prediction[0] > prediction[3]:
        ind=0
        result = "r"
    if prediction[0] < prediction[1] > prediction[2] and prediction[1] > prediction[3]:
        ind=1
        result = "s1"
    if prediction[1] < prediction[2] > prediction[0] and prediction[2] > prediction[3]:
        ind=2
        result = "p"
    if prediction[1] < prediction[3] > prediction[0] and prediction[3] > prediction[2]:
        ind = 3
        result = "s2"



    ptime = ctime
    ctime = time.time()
    fps = 1 / (ctime - ptime)

    cv2.putText(img, result, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
