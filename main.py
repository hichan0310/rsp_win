import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor as tensor

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

x_train_r = np.zeros((1000, 20, 2))
x_train_s = np.zeros((1000, 20, 2))
x_train_p = np.zeros((1000, 20, 2))

print("바위")

for _ in range(1000):
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

    x_train_r[_] = info

    cv2.imshow("Image", img)
    cv2.waitKey(1)
time.sleep(5)

print("가위")

for _ in range(1000):
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

    x_train_s[_] = info

    cv2.imshow("Image", img)
    cv2.waitKey(1)
time.sleep(5)

print("보")

for _ in range(1000):
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

    x_train_p[_] = info

    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

print(x_train_s)
print(x_train_r)
print(x_train_p)
print("data load complete")

x_train_s = tensor(x_train_s.reshape((1000, -1)))
x_train_r = tensor(x_train_r.reshape((1000, -1)))
x_train_p = tensor(x_train_p.reshape((1000, -1)))
print("tensor change complete")



class learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
        )
        self.dropout=nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer(x)
        return x
model=learning()
optimizer=optim.SGD(model.parameters(), lr=0.001)




################################################################
# 이거 반복
x =     # 미니 배치를 어떻게 잘 만들어야 할지 모르겠음
y =
prediction=model(x)
cost = F.cross_entropy(prediction, y)
################################################################



optimizer.zero_grad()
cost.backward()
optimizer.step()

#문제점으로 생각되는 것 : 크기를 일정하게 유지하지 않음
