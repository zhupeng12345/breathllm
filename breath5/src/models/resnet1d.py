import torch
from torch import nn
from torch.nn import functional as F


class ResNet1D(nn.Module):
    def __init__(self, channels, width=64):
        super().__init__()
        self.stem = nn.Conv1d(channels, width, 8, stride=8)
        self.blocks = nn.ModuleList(
            nn.ModuleList([nn.Conv1d(width, width, 5, dilation=d) for _ in range(2)]) for d in (1, 2, 4)
        )
        self.head = nn.Linear(width, 5)

    def forward(self, x):
        z = self.stem(x)
        for block in self.blocks:
            residual = z
            for conv in block:
                z = F.gelu(conv(F.pad(z, (4*conv.dilation[0], 0))))
            z = residual + z
        return self.head(z[:, :, -5:].transpose(1, 2))
