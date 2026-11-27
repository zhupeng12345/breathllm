import torch
from torch import nn
from torch.nn import functional as F


class TCN(nn.Module):
    def __init__(self, channels, width=48):
        super().__init__()
        self.stem = nn.Linear(channels * 8, width)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 3, dilation=d) for d in (1, 2, 4, 8, 16))
        self.head = nn.Linear(width, 5)

    def forward(self, x):
        frames = x.unfold(-1, 8, 8).transpose(1, 2).flatten(2)
        z = self.stem(frames).transpose(1, 2)
        for layer in self.blocks:
            z = z + F.gelu(layer(F.pad(z, (2*layer.dilation[0], 0))))
        return self.head(z[:, :, -5:].transpose(1, 2))
