'''
퓨리케마 보고서
싸다한테 다 털렸죠?
(건욱이의 요청으로 넣었습니다)

이제 활성화 함수 잘 만들고 잘 변수들 잘 조작하면 쉽게 만들 수 있을 것 같아요
'''

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor as tensor
from torch.utils.data import TensorDataset, DataLoader
import csv

sample_size = 1000
batch_size = 20

x_train=[]
with open('x_train.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    i=0
    for line in rdr:
        x_train.append([*map(int, line)])
        i+=1
x_train=tensor(x_train)

y_train=[]
with open('y_train.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    i=0
    for line in rdr:
        y_train.append([*map(int, line)])
        i+=1
y_train=tensor(y_train)


class learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Linear(40, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
model=learning()
optimizer=optim.SGD(model.parameters(), lr=0.001)

datasets=TensorDataset(x_train, y_train)
dataloader=DataLoader(datasets, batch_size=batch_size, shuffle=True)

for epoch in range(501):
    cost_sum = 0
    for batch_ind, sample in enumerate(dataloader):
        x, y=sample
        prediction = model(x)
        cost = F.cross_entropy(prediction, y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        cost_sum += cost.item()
    if epoch%10==0:
        print("epoch {} : {:.5f}".format(epoch, cost_sum/int(sample_size/batch_size)))

    #문제점으로 생각되는 것 : 크기를 일정하게 유지하지 않음

