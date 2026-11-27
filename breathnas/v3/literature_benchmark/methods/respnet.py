"""RespNet ApneaDetectNet with parallel pooling and bidirectional recurrence."""

import torch
from torch import nn

from methods._pool import mean_bins


class ApConvBlock(nn.Module):
    def __init__(self, source, width, kernel, pool):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv1d(source, width, kernel, padding=kernel//2), nn.LeakyReLU(.1),
                                    nn.BatchNorm1d(width), nn.MaxPool1d(pool))
        self.skip = nn.MaxPool1d(pool)

    def forward(self, x):
        return torch.cat((self.branch(x), self.skip(x)), dim=1)


class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        n = len(mode["channels"])
        self.conv = nn.Sequential(
            nn.Conv1d(n, 4, 3, padding=1), nn.LeakyReLU(.1), nn.BatchNorm1d(4), nn.MaxPool1d(2),
            ApConvBlock(4, 8, 5, 2), ApConvBlock(12, 16, 5, 2), ApConvBlock(28, 32, 3, 3),
        )
        self.rnn = nn.LSTM(60, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128+60*5, 128), nn.LeakyReLU(.1), nn.Dropout(.15), nn.Linear(128, 4))

    def forward(self, x):
        features = self.conv(x)
        _, (state, _) = self.rnn(features.transpose(1, 2))
        pooled = mean_bins(features, 5).flatten(1)
        return self.head(torch.cat((state.transpose(0, 1).flatten(1), pooled), dim=1))
