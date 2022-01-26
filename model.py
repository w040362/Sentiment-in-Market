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
        self.fc2 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        c_0 = h_0

        r_out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach()))
        out = self.fc1(r_out[:, -1, :])
        out = F.relu(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        pred_y = self.forward(x)
        return pred_y

