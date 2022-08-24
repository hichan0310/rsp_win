'''
데이터셋 만들어주는 코드
여기에 다른 설명 입력
건욱이 바보
'''

import cv2
import mediapipe as mp
import time
import numpy as np
from torch import FloatTensor as tensor


num_samples = 1000


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

x_train_r = np.zeros((num_samples, 20, 2))
x_train_s1 = np.zeros((num_samples, 20, 2))
x_train_s2 = np.zeros((num_samples, 20, 2))
x_train_p = np.zeros((num_samples, 20, 2))

print("바위")
for _ in range(num_samples):
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

    cv2.putText(img, str(int(_/num_samples*100)), (O_y, O_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
time.sleep(5)

print("가위1")
for _ in range(num_samples):
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

    x_train_s1[_] = info

    cv2.putText(img, str(int(_/num_samples*100)), (O_y, O_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
time.sleep(5)

print("보")
for _ in range(num_samples):
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

    cv2.putText(img, str(int(_/num_samples*100)), (O_y, O_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
time.sleep(5)

print("가위2")
for _ in range(num_samples):
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

    x_train_s2[_] = info

    cv2.putText(img, str(int(_/num_samples*100)), (O_y, O_x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()


x_train_s1 = tensor(x_train_s1.reshape((num_samples, -1)))
x_train_s2 = tensor(x_train_s2.reshape((num_samples, -1)))
x_train_r = tensor(x_train_r.reshape((num_samples, -1)))
x_train_p = tensor(x_train_p.reshape((num_samples, -1)))
print("tensor change complete")

x_train_all=tensor(np.vstack([x_train_r, x_train_s1, x_train_p, x_train_s2]))    # (4*num_samples, 40)
y_train_all=tensor(np.vstack([np.repeat([1, 0, 0, 0], num_samples), np.repeat([0, 1, 0, 0], num_samples), np.repeat([0, 0, 1, 0], num_samples), np.repeat([0, 0, 0, 1], num_samples)]).T)

print(x_train_all.shape)
print(y_train_all.shape)

import csv

with open('x_train.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    for data in x_train_all:
        wr.writerow(data.int().detach().numpy())
with open('y_train.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    for data in y_train_all:
        wr.writerow(data.int().detach().numpy())