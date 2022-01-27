import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import NaivePredictor, BiPredictor


EPOCH = 200
WIDTH = 20
BATCH_SIZE = 16
INPUT_SIZE = 2
HIDDEN_SIZE = 16
NUM_LAYERS = 2
lr = 0.02


def standardize(data, ss=None):
    # for training
    ss = StandardScaler()
    s_x = ss.fit_transform(data)
    return s_x, ss


def extract_pair(data, width):
    length = data.shape[0]
    x = []
    y = []
    for t in range(length-width-1):
        x.append(data[t:t+width])
        y.append([data[t+width][0]])  # 0 dimension: target
    return np.array(x), np.array(y)


def differential(src):
    tmp = src.copy()
    tmp = np.delete(tmp, 0, axis=0)
    src = np.delete(src, -1, axis=0)
    diff = tmp - src
    return diff


path = 'data/googl.us.txt'
df = pd.read_csv(path, sep=',')
# from 2014-8-19 to 2017-11-10
# 3333 in total (Date, Open, High, Low, Close, Volume, OpenInt)
# print(df)

# select Close data as example
data_close = df['Close'].values.reshape(-1, 1)
data_open = df['Open'].values.reshape(-1, 1)
# print(data_close)
# plt.plot(data_close, '-')
# plt.show()

# data_close_diff = differential(data_close)
# data_close_diff_, ss_diff = standardize(data_close_diff)
# data_ = data_close_diff_

data_close_, ss_close = standardize(data_close)
data_open_, ss_open = standardize(data_open)
data_ = np.hstack((data_close_, data_open_))        # input_size = 2

x, y = extract_pair(data_, WIDTH)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(data_.shape)
# print(x.shape)
x_train = torch.tensor(x_train, dtype=torch.float32)    # (total, width, input_size=?) !
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# print(x_train.shape)
# print(y_train.shape)
data_train = TensorDataset(x_train, y_train)
data_loader = DataLoader(data_train, BATCH_SIZE, shuffle=True)
# print(data_train[0:2])

# ------------- train -------------
model = NaivePredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
# model = BiPredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

optimizer = torch.optim.Adam(model.parameters(), lr)
loss_func = nn.MSELoss()

loss_list = []

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(data_loader):
        # print(b_x.shape)
        # print(b_y.shape)
        # b_x.shape (batch_size, width, input_size)
        out = model(b_x)
        # print(out.shape)  # (batch_size, input_size)
        # print(b_y.shape)
        loss = loss_func(out, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print('epoch: {} | step: {} | loss: {}'.format(epoch, step, loss))
            loss_list.append(loss)

model_path = './model_.pth'
torch.save(model, model_path)

