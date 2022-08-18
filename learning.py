"""
퓨리케마 보고서
싸다한테 다 털렸죠?
(건욱이의 요청으로 넣었습니다)

이제 활성화 함수 잘 만들고 잘 변수들 잘 조작하면 쉽게 만들 수 있을 것 같아요
"""

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor as tensor
from torch.utils.data import TensorDataset, DataLoader
import csv

sample_size = 1000
batch_size = 20
epoches = 200

x_train = []
with open('x_train.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    i = 0
    for line in rdr:
        x_train.append([*map(int, line)])
        i += 1
x_train = tensor(x_train)

y_train = []
with open('y_train.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    i = 0
    for line in rdr:
        y_train.append([*map(int, line)])
        i += 1
y_train = tensor(y_train)


class learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(40, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 3)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer(x)
        return x


model = learning()
optimizer = optim.SGD(model.parameters(), lr=0.001)

datasets = TensorDataset(x_train, y_train)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

for epoch in range(epoches + 1):
    cost_sum = 0
    for batch_ind, sample in enumerate(dataloader):
        x, y = sample
        prediction = model(x)
        cost = F.cross_entropy(prediction, y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        cost_sum += cost.item()
    if epoch % 10 == 0:
        print("epoch {} : {:.5f}".format(epoch, cost_sum / int(sample_size / batch_size)))

    # 문제점으로 생각되는 것 : 크기를 일정하게 유지하지 않음

import cv2
import numpy as np
import mediapipe as mp
import time

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

    result = "none"
    ind=-1
    if prediction[1]<prediction[0]>prediction[2]:
        ind=0
        result = "r"
    if prediction[0]<prediction[1]>prediction[2]:
        ind=1
        result = "s"
    if prediction[1]<prediction[2]>prediction[0]:
        ind=2
        result = "p"



    ptime = ctime
    ctime = time.time()
    fps = 1 / (ctime - ptime)

    cv2.putText(img, result, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
