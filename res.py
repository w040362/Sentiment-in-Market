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
        y.append(data[t+width])
    return np.array(x), np.array(y)


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

data_, ss = standardize(data_close)
x, y = extract_pair(data_, WIDTH)
x = torch.tensor(x, dtype=torch.float32)

# ------------- evaluate -------------

model_path = './model.pth'
model = torch.load(model_path)
predict = model(x)

predict = predict.detach().numpy()

predict = ss.inverse_transform(predict)
y = ss.inverse_transform(y)

# red for real, blue for prediction
plt.plot(y, 'r-')
plt.plot(predict, 'b-')
plt.show()

