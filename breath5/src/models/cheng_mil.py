import torch
from torch import nn
from torch.nn import functional as F


class ChengMIL(nn.Module):
    def __init__(self, channels, classes=5, width=48):
        super().__init__()
        self.stem = nn.Linear(channels * 8, width)
        self.conv = nn.ModuleList(nn.Conv1d(width, width, k) for k in (3, 5, 9))
        self.attention = nn.Linear(width, 1)
        self.head = nn.Linear(2 * width, classes)

    def forward(self, signal):
        frames = signal.unfold(-1, 8, 8).transpose(1, 2).flatten(2)
        z = self.stem(frames).transpose(1, 2)
        z = sum(F.gelu(c(F.pad(z, (c.kernel_size[0]-1, 0)))) for c in self.conv).transpose(1, 2) / 3
        score = self.attention(z).squeeze(-1)
        allowed = torch.arange(60, device=z.device)[None, :] <= torch.arange(55, 60, device=z.device)[:, None]
        weight = score[:, None].masked_fill(~allowed, -torch.inf).softmax(-1)
        aggregate = weight @ z
        return self.head(torch.cat((aggregate, z[:, -5:]), -1))
