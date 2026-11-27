"""Minute-scale statistical features and an RBF SVM."""

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def features(x, hz):
    x = x.reshape(*x.shape[:2], -1, hz).mean(axis=-1, dtype=np.float64)
    mean = x.mean(2)
    demean = x - mean[:, :, None]
    t = np.arange(x.shape[2]) - (x.shape[2]-1)/2
    dot = (demean*t).sum(2)
    slope = dot/(t@t)
    corr = dot/np.sqrt((t@t)*(demean**2).sum(2)).clip(1e-8)
    crossing = ((demean[:, :, 1:] >= 0) != (demean[:, :, :-1] >= 0)).sum(2)
    delta = np.abs(np.diff(x.reshape(*x.shape[:2], 5, -1).mean(3), axis=2)).mean(2)
    out = (x.max(2), x.min(2), mean, x.var(2), corr, crossing, slope, np.abs(slope),
           mean-slope*(x.shape[2]-1)/2, delta)
    return np.stack(out, axis=2).reshape(len(x), -1)


class Model:
    def __init__(self, mode, seed, jobs):
        self.mode = mode
        self.params = ((1000, .02), (50, "scale"), (200, .10))
        self.scaler = MinMaxScaler()
        self.models = {}

    def fit(self, x, y, windows, weights):
        view = self.scaler.fit_transform(features(x, self.mode["hz"]))
        for name, (cost, gamma) in zip(self.variants(), self.params):
            model = SVC(C=cost, gamma=gamma, cache_size=280, decision_function_shape="ovo")
            model.fit(view, y, sample_weight=weights[y])
            self.models[name] = model

    def variants(self):
        return ("paper_C1000_gamma002", "C50_scale", "C200_gamma01")

    def bases(self, x):
        return self.scaler.transform(features(x, self.mode["hz"]))

    def predict(self, base, name):
        # Use pairwise margins, retaining strength of evidence instead of OVR votes.
        pairwise = self.models[name].decision_function(base)
        logits = np.zeros((len(base), 4))
        for j, (left, right) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
            logits[:, left] += pairwise[:, j]
            logits[:, right] -= pairwise[:, j]
        return softmax(logits, axis=1)
