"""A compact EEG CNN approximation of the public Mahmud abstract."""

from torch import nn

from methods._pool import mean_bins


class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        stride = max(1, mode["hz"]//16)
        self.body = nn.Sequential(
            nn.Conv1d(len(mode["channels"]), 32, 2*stride+7, stride=stride, padding=stride+3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 7, padding=3), nn.BatchNorm1d(96), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(576, 96), nn.GELU(), nn.Dropout(.15), nn.Linear(96, 4))

    def forward(self, x):
        return self.head(mean_bins(self.body(x), 6))
