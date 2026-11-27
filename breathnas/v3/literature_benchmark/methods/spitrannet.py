"""Convolutional tokenizer with surrogate-spike encoder and decoder."""

import torch
from torch import nn


def spike(x):
    soft = torch.sigmoid(2 * (x - .3))
    return (x >= .3).to(x.dtype) + (soft - soft.detach())


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, 2*width)
        self.second = nn.Linear(2*width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(.1)

    def forward(self, query, memory):
        mixed = self.attention(query, memory, memory, need_weights=False)[0]
        hidden = self.attention_norm(query + self.dropout(spike(mixed)))
        return self.norm(hidden + self.dropout(self.second(spike(self.first(hidden)))))


class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        widths = [len(mode["channels"]), 32, 64, 64]
        stride = max(1, mode["hz"]//32)
        layers = []
        for i in range(3):
            layers.extend((nn.Conv1d(widths[i], widths[i+1], 17, stride=stride if i == 0 else 1, padding=8),
                           nn.BatchNorm1d(widths[i+1]), nn.ReLU(), nn.MaxPool1d(2)))
        self.cnn = nn.Sequential(*layers)
        angles = torch.arange(64)[:, None] / (10000 ** (torch.arange(0, 64, 2)/64))
        self.register_buffer("position", torch.stack((angles.sin(), angles.cos()), -1).flatten(1))
        self.encoder = Block(64)
        self.decoder_self = Block(64)
        self.decoder_cross = Block(64)
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        local = self.cnn(x).transpose(1, 2)
        local = local + self.position[:local.shape[1]]
        memory = self.encoder(local, local)
        decoded = self.decoder_self(local, local)
        return self.head(self.decoder_cross(decoded, memory).mean(1))
