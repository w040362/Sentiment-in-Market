import torch
import torch.nn as nn
import torch.nn.functional as F


class NaivePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, drop_prob=0.):
        super(NaivePredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=drop_prob
        )

        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn1 = nn.BatchNorm1d(int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)

        r_out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach()))
        # print(r_out.shape)    (batch_size, width, hidden_size)
        out = self.fc1(r_out[:, -1, :])
        out = F.relu(out)
        out = self.fc2(out)
        # print(out.shape)
        return out

    def predict(self, x):
        pred_y = self.forward(x)
        return pred_y


class BiPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, drop_prob=0.):
        super(BiPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=drop_prob,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size)

        r_out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach()))
        # print(r_out.shape)    (batch_size, width, hidden_size)
        out = self.fc1(r_out[:, -1, :])
        out = F.relu(out)
        out = self.fc2(out)
        # print(out.shape)
        return out

    def predict(self, x):
        pred_y = self.forward(x)
        return pred_y


# textCNN hyper_parameters
embed_size = 768
num_classes = 2     # positive or negative (1/0)
max_length = 256


class textCNN(nn.Module):
    def __init__(self, output_channel=3):
        super(textCNN, self).__init__()
        self.output_channel = output_channel

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.output_channel, kernel_size=(2, embed_size)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # add 1 channel (batch_size, 1, seq_len=256, hidden_size=768)
        batch_size = x.shape[0]

        conv_x = self.conv(x)
        flat_x = conv_x.view(batch_size, -1)    # (batch_size, output_channel)
        out = self.fc(flat_x)
        return out
