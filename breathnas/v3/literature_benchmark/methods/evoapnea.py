"""Local CNN, causal Transformer and waveform statistics, with factored outputs."""

import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        channels = len(mode["channels"])
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 64, 25, padding=12), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 25, padding=12), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(.3),
        )
        angles = torch.arange(44)[:, None] / (10000 ** (torch.arange(0, 128, 2)/128))
        self.register_buffer("position", torch.stack((angles.sin(), angles.cos()), -1).flatten(1))
        self.register_buffer("causal", torch.triu(torch.ones(44, 44, dtype=torch.bool), diagonal=1))
        layer = nn.TransformerEncoderLayer(128, 4, 128, batch_first=True, dropout=.1)
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.traits = nn.Sequential(nn.Linear(8 * channels, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 32), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(288, 256), nn.ReLU(), nn.LayerNorm(256),
                                  nn.Dropout(.3), nn.Linear(256, 128), nn.ReLU())
        self.detect = nn.Linear(128, 1)
        self.subtype = nn.Linear(128, 2)
        self.background = nn.Linear(128, 2)

    def forward(self, x):
        wave = F.interpolate(x, size=88, mode="linear", align_corners=False)
        local = self.conv(wave).transpose(1, 2)
        order = self.temporal(local + self.position, mask=self.causal, is_causal=True)
        sections = wave.reshape(len(wave), wave.shape[1], 4, 22)
        amplitude = sections.amax(3) - sections.amin(3)
        stats = torch.cat((amplitude, wave.mean(2, keepdim=True), wave.std(2, keepdim=True),
                           wave.diff(dim=2).abs().mean(2, keepdim=True), wave[:, :, -1:] - wave[:, :, :1]), dim=2)
        h = self.fuse(torch.cat((local.mean(1), order.mean(1), self.traits(stats.flatten(1))), dim=1))
        detector = self.detect(h)
        pos = F.logsigmoid(detector) + self.subtype(h).log_softmax(1)
        neg = F.logsigmoid(-detector) + self.background(h).log_softmax(1)
        return torch.cat((neg[:, :1], pos, neg[:, 1:]), dim=1)
