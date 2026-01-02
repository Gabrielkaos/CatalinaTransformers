import torch
import torch.nn as nn
import math


class RNN(nn.Module):
    def __init__(self, n_input, n_hiddens, num_layers, n_output, device=None):
        super(RNN, self).__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.n_hidden = n_hiddens
        self.n_layers = num_layers
        self.embed = nn.Embedding(n_input, n_hiddens).to(self.device)
        self.lstm = nn.LSTM(n_hiddens, n_hiddens, num_layers, batch_first=True, bidirectional=True).to(self.device)

        self.fc = nn.Linear(n_hiddens * 2, n_output).to(self.device)

    def forward(self, x, hiddens, cells):
        output_n = self.embed(x)

        output_n, (hiddens, cells) = self.lstm(output_n.unsqueeze(1), (hiddens, cells))

        output_n = self.fc(output_n.reshape(output_n.shape[0], -1))
        return output_n, (hiddens, cells)

    def init_hidden(self, batch_num):
        hiddens = torch.zeros(self.n_layers * 2, batch_num, self.n_hidden).to(self.device)
        cells = torch.zeros(self.n_layers * 2, batch_num, self.n_hidden).to(self.device)
        return hiddens, cells
