import torch
from torch import nn
import torch.nn.functional as F


class AttnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, time_step):
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        self.attn2 = nn.Linear(in_features=self.T, out_features=self.T)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)
        #self.attn = nn.Sequential(attn1, attn2, nn.Tanh(), attn3)

    def forward(self, driving_x):
        batch_size = driving_x.size(0)
        # print(driving_x.shape)  # (batch_size, time_step, 1)

        # batch_size * time_step * hidden_size
        code = torch.zeros((batch_size, self.T, self.hidden_size))
        # initialize hidden state
        h = torch.zeros((1, batch_size, self.hidden_size))
        # initialize cell state
        s = torch.zeros((1, batch_size, self.hidden_size))
        if torch.cuda.is_available():
            code = code.cuda()
            h = h.cuda()
            s = s.cuda()

        for t in range(self.T):
            x = torch.cat((self.embedding_hidden(h), self.embedding_hidden(s)),
                          dim=2)                            # batch_size * input_size * (2 * hidden_size)
            z1 = self.attn1(x)                              # (bs, input_size, time_step)
            z2 = self.attn2(driving_x.permute(0, 2, 1))     # (bs, 1, time_step) -> (bs, 1, time_step)
            x = z1 + z2                                     # (bs, input_size, time_step)

            z3 = self.attn3(self.tanh(x))       # batch_size * input_size * 1
            attn_w = F.softmax(z3.view(batch_size, -1), dim=1)

            # batch_size * input_size
            weighted_x = torch.mul(attn_w, driving_x[:, t, :])
            _, states = self.lstm(weighted_x.unsqueeze(0), (h, s))
            h = states[0]
            s = states[1]

            # encoding result
            # batch_size * time_step * encoder_hidden_size
            code[:, t, :] = h

        return code

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)


class AttnDecoder(nn.Module):
    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h, y_seq):
        batch_size = h.size(0)

        d = torch.zeros((1, batch_size, self.hidden_size))
        s = torch.zeros((1, batch_size, self.hidden_size))
        ct = torch.zeros((batch_size, self.hidden_size))
        if torch.cuda.is_available():
            d = d.cuda()
            s = s.cuda()
            ct = ct.cuda()

        for t in range(self.T):
            # batch_size * time_step * (encoder_hidden_size + decoder_hidden_size)
            x = torch.cat((self.embedding_hidden(d), self.embedding_hidden(s)), 2)
            z1 = self.attn1(x)
            z2 = self.attn2(h)              # batch_size * time_step * hidden_size
            x = z1 + z2                     # batch_size * time_step * hidden_size

            z3 = self.attn3(self.tanh(x))   # batch_size * time_step * 1
            beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            # print(beta_t.shape)   # batch_size * time_step

            ct = torch.bmm(beta_t.unsqueeze(1), h).squeeze(1)   # ?
            if t < self.T - 1:
                yc = torch.cat((y_seq[:, t].unsqueeze(1), ct), dim=1)
                y_tilde = self.tilde(yc)
                _, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
                d = states[0]
                s = states[1]
        # batch_size * 1
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), ct), dim=1)))
        return y_res

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)
