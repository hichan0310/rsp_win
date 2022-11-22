import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor as tensor
from torch.utils.data import TensorDataset, DataLoader
import csv
import numpy as np
import matplotlib.pyplot as plt

sample_size = 260
batch_size = 10
epoches = 10000

x_train = []
with open('x_train.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    i = 0
    for line in rdr:
        x_train.append([*map(float, line)])
        i += 1
x_train = tensor(x_train)

y_train = []
with open('y_train.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    i = 0
    for line in rdr:
        y_train.append([*map(float, line)])
        i += 1
y_train = tensor(y_train)


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
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.layer(x)
        return x


model = learning()
optimizer = optim.SGD(model.parameters(), lr=0.001)

datasets = TensorDataset(x_train, y_train)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

arr = []
for epoch in range(epoches + 1):
    cost_sum = 0
    for batch_ind, sample in enumerate(dataloader):
        x, y = sample
        prediction = model(x)
        cost = F.binary_cross_entropy(prediction, y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        cost_sum += cost.item()
    if epoch % 10 == 0:
        print("epoch {} : {:.5f}".format(epoch, cost_sum / int(sample_size / batch_size)))
    arr.append(cost_sum / int(sample_size / batch_size))

arr = np.array(arr)
plt.plot(np.arange(len(arr)), arr, color="black", label="cost")
plt.title('Training Loss')
plt.show()
torch.save(model.state_dict(), 'regression_model.pth')
