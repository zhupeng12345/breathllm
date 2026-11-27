"""Coarse and central ten-second random-forest detection."""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier


def features(x, mode, coarse):
    hz = mode["hz"]
    result = []
    for i, channel in enumerate(mode["channels"]):
        wave = x[:, i]
        if channel == "spo2":
            wave = wave.clip(50, 100)
            reference = x[:, i, :max(hz, wave.shape[1]//3)].clip(50, 100)
            baseline = np.stack((reference.max(1), reference.mean(1)), 1)
            std, span = wave.std(1), np.ptp(wave, axis=1)
            result.extend((std, span, *(wave[:, None] < .98*baseline[:, :, None]).mean(2).T))
            if not coarse:
                result.extend((wave[:, -1]-wave[:, 0], *(baseline-wave.mean(1)[:, None]).T,
                               (wave < 92).mean(1), (wave < 91).mean(1), (x[:, i] < 50).mean(1)))
        else:
            wave = sosfiltfilt(butter(3, .05, "highpass", fs=hz, output="sos"),
                               uniform_filter1d(wave, size=4, axis=1), axis=1)
            span = maximum_filter1d(wave, 3*hz, axis=1)-minimum_filter1d(wave, 3*hz, axis=1)
            amplitude = span[:, hz::2*hz]
            ref = amplitude.max(1).clip(1e-8)
            relative = amplitude/ref[:, None]
            result.extend(((relative < .7).sum(1), (relative < .7).mean(1)))
            if not coarse:
                frequency = np.fft.rfftfreq(wave.shape[1], 1 / hz)
                spectrum = np.abs(np.fft.rfft(wave, axis=1))[:, (frequency >= .2) & (frequency <= .4)]
                spectrum = spectrum / spectrum.sum(1, keepdims=True).clip(1e-8)
                result.extend((amplitude.mean(1), amplitude.std(1), np.ptp(amplitude, axis=1),
                               (relative < .3).sum(1), (relative < .3).mean(1), (relative > .85).sum(1),
                               (relative > .85).mean(1), (spectrum**4).sum(1)))
    return np.column_stack(result).astype(np.float32)


def central_features(x, mode):
    end = (20 - mode["offset"]) * mode["hz"]
    return features(x[:, :, end-10*mode["hz"]:end], mode, False)


class Model:
    def __init__(self, mode, seed, jobs):
        self.mode = mode
        kwargs = dict(random_state=seed, n_jobs=jobs, min_samples_leaf=2)
        self.coarse = RandomForestClassifier(n_estimators=64, **kwargs)
        self.fine = RandomForestClassifier(n_estimators=128, **kwargs)

    def fit(self, x, y, windows, weights):
        self.coarse.fit(features(x, self.mode, True), y, sample_weight=weights[y])
        blocks, labels = [], []
        hz, offset = self.mode["hz"], self.mode["offset"]
        for i in range(3):
            a, b = (10*i - offset)*hz, (10*(i+1) - offset)*hz
            target = windows[f"fine{i}"].to_numpy()
            valid = target >= 0
            blocks.append(features(x[valid, :, a:b], self.mode, False))
            labels.append(target[valid])
        target = np.concatenate(labels)
        self.fine.fit(np.concatenate(blocks), target, sample_weight=weights[target])
        assert list(self.coarse.classes_) == list(self.fine.classes_) == list(range(4))

    def variants(self):
        return ("coarse_0.75", "coarse_0.50", "coarse_0.25")

    def bases(self, x):
        return self.coarse.predict_proba(features(x, self.mode, True)), self.fine.predict_proba(central_features(x, self.mode))

    def predict(self, base, name):
        weight = float(name.split("_")[1])
        coarse, fine = base
        # Use the coarse detector as a soft gate and keep both subtype estimates.
        pc = coarse[:, 1:3].sum(1).clip(1e-6, 1-1e-6)
        pf = fine[:, 1:3].sum(1).clip(1e-6, 1-1e-6)
        probability = (1-weight)*fine + weight*coarse
        odds = (pc/(1-pc))**weight * (pf/(1-pf))**(1-weight)
        target = odds/(1+odds)
        for indices, total in (([1, 2], target), ([0, 3], 1-target)):
            normalized = probability[:, indices] + 1e-8
            probability[:, indices] = total[:, None] * normalized / normalized.sum(1, keepdims=True)
        return probability
