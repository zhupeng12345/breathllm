"""Layers shared by convolutional UCDDB baselines."""

from torch import nn


def convolution(inputs, outputs, kernel, activation=True):
    layers = [nn.Conv1d(inputs, outputs, kernel, padding="same", bias=False),
              nn.BatchNorm1d(outputs)]
    if activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def temporal_classifier(body, channels):
    return nn.Sequential(body, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, 3))
