"""InceptionTime network for each ensemble member."""

import torch
from torch import nn

from ._layers import convolution, temporal_classifier


class InceptionModule(nn.Module):
    def __init__(self, inputs, filters=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(inputs, filters, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv1d(filters, filters, kernel, padding="same", bias=False)
            for kernel in (40, 20, 10)
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(inputs, filters, 1, bias=False),
        )
        self.norm = nn.BatchNorm1d(filters * 4)

    def forward(self, signal):
        bottleneck = self.bottleneck(signal)
        branches = [branch(bottleneck) for branch in self.branches]
        branches.append(self.pool_branch(signal))
        return torch.relu(self.norm(torch.cat(branches, dim=1)))


class InceptionStack(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(InceptionModule(inputs),
                                    InceptionModule(128), InceptionModule(128))
        self.skip = convolution(inputs, 128, 1, activation=False)

    def forward(self, signal):
        return torch.relu(self.layers(signal) + self.skip(signal))


def build_inceptiontime(inputs):
    return temporal_classifier(nn.Sequential(InceptionStack(inputs), InceptionStack(128)), 128)
