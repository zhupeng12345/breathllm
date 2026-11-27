from torch import nn
from torch.nn import functional as F


class YuLSTMCNN(nn.Module):
    def __init__(self, channels, classes=5, width=64):
        super().__init__()
        self.lstm = nn.LSTM(channels * 8, width, batch_first=True)
        self.conv = nn.Conv1d(width, width, 5)
        self.head = nn.Linear(width, classes)

    def forward(self, signal):
        frames = signal.unfold(-1, 8, 8).transpose(1, 2).flatten(2)
        z, _ = self.lstm(frames)
        z = F.gelu(self.conv(F.pad(z.transpose(1, 2), (4, 0)))).transpose(1, 2)
        return self.head(z[:, -5:])
