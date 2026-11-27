"""Frequency-band energy ratios with cosine K-nearest neighbors."""

from itertools import combinations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def features(x, mode):
    eeg = mode["channels"] == ["c3", "c4"]
    if eeg:
        x = x.mean(1, keepdims=True)
        bands = [[.25, 4, 8, 12, 16, 40.00001]]
    else:
        bands = [[.066, .15, .25, .40, .65, 1.51] if c != "spo2" else [.033, .08, .14, .24, .45, 1.51]
                 for c in mode["channels"]]
    x = x - x.mean(2, keepdims=True)
    x = x / np.maximum(x.max(2, keepdims=True), 1e-8)
    power = np.abs(np.fft.rfft(x, axis=2))**2
    hz = mode["hz"]
    frequency = np.fft.rfftfreq(x.shape[2], 1/hz)
    result = []
    for i, edges in enumerate(bands):
        energy = np.column_stack([power[:, i, (frequency >= a) & (frequency < b)].sum(1)
                                  for a, b in zip(edges, edges[1:])])
        energy += np.maximum(1e-8, energy.sum(1, keepdims=True)*1e-7)
        ratio = np.column_stack([energy[:, a]/energy[:, b] for a, b in combinations(range(5), 2)])
        result.append(normalize(ratio.astype(np.float32)))
    return np.ascontiguousarray(normalize(np.concatenate(result, axis=1)), dtype=np.float32)


class Model:
    def __init__(self, mode, seed, jobs):
        self.model = NearestNeighbors(n_neighbors=41, metric="cosine", algorithm="brute", n_jobs=jobs)
        self.mode = mode

    def fit(self, x, y, windows, weights):
        self.model.fit(features(x, self.mode))
        self.labels = y
        self.weights = np.sqrt(weights)

    def variants(self):
        return ("k5", "k17", "k41")

    def bases(self, x):
        distance, ids = self.model.kneighbors(features(x, self.mode))
        return np.stack((distance, self.labels[ids]), axis=-1)

    def predict(self, base, name):
        k = int(name[1:])
        distance, labels = base[:, :k].transpose(2, 0, 1)
        weight = 1 / (distance.clip(0, 2) + .04)
        votes = np.stack([(weight * (labels == i)).sum(1) + .01 for i in range(4)], axis=1) * self.weights
        return votes/votes.sum(1, keepdims=True)
