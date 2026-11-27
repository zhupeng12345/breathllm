import torch
from torch import nn


class CnnLstmTeacher(nn.Module):
    def __init__(self, channels=4, width=64):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(channels, width, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.history = nn.LSTM(width, width, batch_first=True)
        self.head = nn.Linear(width, 5)

    def forward(self, signal):
        batch, channels, samples = signal.shape
        frames = signal.reshape(batch, channels, samples // 8, 8).transpose(1, 2).reshape(-1, channels, 8)
        features = self.frame_encoder(frames).reshape(batch, -1, self.history.input_size)
        states, _ = self.history(features)
        return self.head(states[:, -5:])
