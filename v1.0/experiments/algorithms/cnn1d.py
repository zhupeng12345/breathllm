"""One-dimensional convolutional baseline."""

from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(nn.Conv1d(4,32,9,padding=4),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(4),nn.Conv1d(32,64,7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(4),nn.Conv1d(64,96,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.fc=nn.Linear(96,3)
    def forward(self,x): return self.fc(self.body(x).squeeze(-1))
