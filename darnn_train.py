import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
from darnn_model import AttnEncoder, AttnDecoder

import math


ENCODER_HIDDEN_SIZE = 64
DECODER_HIDDEN_SIZE = 64
SENTIMENT = 'data/googl.us.csv'     # x_input
STOCK = 'data/msft.us.csv'         # y_input and target

# device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset:
    def __init__(self, sent_csv, target_csv, T, split_ratio=0.8, normalized=False):
        sent_data = pd.read_csv(sent_csv, sep=',')
        stock_data = pd.read_csv(target_csv, sep=',')
        # 保持数据长短一致
        if sent_data.shape[0] > stock_data.shape[0]:
            sent_data = self.crop(sent_data, stock_data['Date'][0]).reset_index()
        else:
            stock_data = self.crop(stock_data, sent_data['Date'][0]).reset_index()
        sent_data = sent_data['Close'].fillna(method='pad')     # change: sentiment score
        stock_data = stock_data['Close'].fillna(method='pad')

        self.train_size = int(split_ratio * (stock_data.shape[0] - T - 1))
        self.test_size = stock_data.shape[0] - T - 1 - self.train_size
        if normalized:
            stock_data = stock_data - stock_data.mean()
        self.x_input, self.y_input, self.y_target = self.time_series_gen(sent_data, stock_data, T)
        # self.X = self.percent_normalization(self.X)
        # self.y = self.percent_normalization(self.y)
        # self.y_seq = self.percent_normalization(self.y_seq)

    def get_size(self):
        return self.train_size, self.test_size

    def get_num_features(self):
        return self.x_input.shape[1]

    def get_train_set(self):
        return self.x_input[:self.train_size], self.y_input[:self.train_size], self.y_target[:self.train_size]

    def get_test_set(self):
        return self.x_input[self.train_size:], self.y_input[self.train_size:], self.y_target[self.train_size:]

    def time_series_gen(self, X, y, T):
        x_input, y_input, y_target = [], [], []
        for i in range(len(X) - T - 1):
            x_input.append(X[i: i+T])
            y_input.append(y[i: i+T])
            y_target.append(y[i+T])
        return np.array(x_input), np.array(y_input), np.array(y_target)

    def crop(self, df, date):
        start = df.loc[df['Date'] == date].index[0]
        return df[start:]

    def log_normalization(self, X):
        X_norm = np.zeros(X.shape[0])
        X_norm[0] = 0
        for i in range(1, X.shape[0]):
            X_norm[i] = math.log(X[i] / X[i-1])
        return X_norm

    def percent_normalization(self, X):
        if len(X.shape) == 2:
            X_norm = np.zeros((X.shape[0], X.shape[1]))
            for i in range(1, X.shape[0]):
                X_norm[i, 0] = 0
                X_norm[i] = np.true_divide(X[i] - X[i-1], X[i-1])
        else:
            X_norm = np.zeros(X.shape[0])
            X_norm[0] = 0
            for i in range(1, X.shape[0]):
                X_norm[i] = (X[i] - X[i-1]) / X[i]
        return X_norm


class TrainDataset(Data.Dataset):
    def __init__(self, x, y_seq, y_tar=None, with_label=True):
        self.x = x
        self.y_seq = y_seq
        self.y_tar = y_tar
        self.with_label = with_label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y_seq = self.y_seq[idx]
        y_tar = self.y_tar[idx]
        if self.with_label:
            return x, y_seq, y_tar
        else:
            return x, y_seq


class Trainer:
    def __init__(self, sent, stock, time_step, split, lr):
        self.dataset = Dataset(sent, stock, time_step, split)
        self.encoder = AttnEncoder(input_size=self.dataset.get_num_features(),
                                   hidden_size=ENCODER_HIDDEN_SIZE, time_step=time_step).to(device)
        self.decoder = AttnDecoder(code_hidden_size=ENCODER_HIDDEN_SIZE,
                                   hidden_size=DECODER_HIDDEN_SIZE, time_step=time_step).to(device)

        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr)
        self.loss_func = nn.MSELoss()
        self.train_size, self.test_size = self.dataset.get_size()

    def train_minibatch(self, num_epochs, batch_size):
        x_input, y_input, y_target = self.dataset.get_train_set()
        train_data = TrainDataset(x_input, y_input, y_target)
        train_dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            loss_sum = 0
            for i, batch in enumerate(train_dataloader):
                var_x, var_y_seq, var_y = [x.to(torch.float32).to(device) for x in batch]

                if var_x.dim() == 2:
                    var_x = var_x.unsqueeze(2)
                var_y = var_y.unsqueeze(1)

                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                # var_x.shape (batch_size, seq_len, 1)
                # var_y.shape (batch_size, 1)
                # var_seq.shape (batch_size, seq_len)
                code = self.encoder(var_x)
                y_res = self.decoder(code, var_y_seq)
                loss = self.loss_func(y_res, var_y)
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                print('[%d], [%d], loss is %f' % (epoch, i, 10000 * loss.item()))
                loss_sum += loss.item()
            print('epoch [%d] finished, the average loss is %f' % (epoch, loss_sum))

            if (epoch + 1) % 100 == 0 or epoch + 1 == num_epochs:
                torch.save(self.encoder.state_dict(), 'models/encoder' + str(epoch + 1) + '-norm' + '.model')
                torch.save(self.decoder.state_dict(), 'models/decoder' + str(epoch + 1) + '-norm' + '.model')

    def test(self, num_epochs, batch_size):
        x_train, y_seq_train, y_train = self.dataset.get_train_set()
        x_test, y_seq_test, y_test = self.dataset.get_test_set()
        y_pred_train = self.predict(x_train, y_seq_train, batch_size)
        y_pred_test = self.predict(x_test, y_seq_test, batch_size)
        plt.figure(figsize=(8,6), dpi=100)
        plt.plot(range(2000, self.train_size), y_train[2000:], label='train truth', color='black')
        plt.plot(range(self.train_size, self.train_size + self.test_size), y_test, label='ground truth', color='black')
        plt.plot(range(2000, self.train_size), y_pred_train[2000:], label='predicted train', color='red')
        plt.plot(range(self.train_size, self.train_size + self.test_size), y_pred_test, label='predicted test', color='blue')
        plt.xlabel('Days')
        plt.ylabel('Stock price')
        plt.savefig('results/res-' + str(num_epochs) + '-' + str(batch_size) + '.png')

    def predict(self, x, y_seq, batch_size):
        y_pred = np.zeros(x.shape[0])
        i = 0
        while i < x.shape[0]:
            batch_end = i + batch_size
            if batch_end > x.shape[0]:
                batch_end = x.shape[0]
            var_x_input = torch.tensor(x[i: batch_end], dtype=torch.float32).to(device)
            var_y_input = torch.tensor(y_seq[i: batch_end], dtype=torch.float32).to(device)

            if var_x_input.dim() == 2:
                var_x_input = var_x_input.unsqueeze(2)
            code = self.encoder(var_x_input)
            y_res = self.decoder(code, var_y_input)
            for j in range(i, batch_end):
                y_pred[j] = y_res[j - i, -1]
            i = batch_end
        return y_pred

    def load_model(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))


def getArgParser():
    parser = argparse.ArgumentParser(description='Train the dual-stage attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=1,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=128,
        help='the mini-batch size')
    parser.add_argument(
        '-s', '--split', type=float, default=0.8,
        help='the split ratio of validation set')
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.01,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='the model name(after encoder/decoder)'
    )
    return parser


if __name__ == '__main__':
    # args = getArgParser().parse_args()
    # num_epochs = args.epoch
    # batch_size = args.batch
    # split = args.split
    # lr = args.lrate
    # test = args.test
    # mname = args.model
    num_epochs = 5
    batch_size = 4
    split = 0.8
    lr = 1e-2
    test = False
    mname = '300'

    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(SENTIMENT, STOCK, 10, split, lr)
    if not test:
        trainer.train_minibatch(num_epochs, batch_size)
    else:
        encoder_name = 'models/encoder' + mname + '-norm.model'
        decoder_name = 'models/decoder' + mname + '-norm.model'
        trainer.load_model(encoder_name, decoder_name)
        trainer.test(mname, batch_size)