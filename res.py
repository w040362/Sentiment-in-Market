import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import NaivePredictor


WIDTH = 25


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
        y.append([data[t+width][0]])
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
data_ = np.hstack((data_close_, data_open_))

x, y = extract_pair(data_, WIDTH)
x = torch.tensor(x, dtype=torch.float32)

# ------------- evaluate -------------

model_path = './model_.pth'
model = torch.load(model_path)
predict = model(x)

predict = predict.detach().numpy()

predict = ss_close.inverse_transform(predict)
y = ss_close.inverse_transform(y)

# red for real, blue for prediction
plt.plot(y, 'r-')
plt.plot(predict, 'b-')
plt.show()

