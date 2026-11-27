"""Compute causal UCDDB features and fit the traditional baseline classifiers."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RATE = 8
HISTORY = 60
CHANNELS = ("Flow", "ribcage", "abdo", "SpO2")
CASCADE_CHANNELS = ("Flow", "SpO2")


def roll_sum(values, width):
    padded = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    t = np.arange(1, len(padded))
    return padded[t] - padded[np.maximum(0, t - width)]


def roll_extreme(values, width, maximum):
    filter_ = maximum_filter1d if maximum else minimum_filter1d
    missing = -np.inf if maximum else np.inf
    return filter_(values, size=width, origin=(width - 1) // 2, mode="constant", cval=missing)


def divide(num, denominator):
    return np.divide(num, denominator, out=np.zeros_like(num, dtype=float), where=denominator > 0)


def variance(sum_x, sum_square, count):
    mean = divide(sum_x, count)
    return np.maximum(divide(sum_square, count) - mean ** 2, 0)


def matrix_features(signal, use_cascade):
    if signal.ndim != 2 or signal.shape[1] % RATE or signal.shape[1] < RATE * HISTORY:
        raise ValueError("Expected a full 8 Hz waveform with at least 60 seconds")
    seconds = signal.reshape(len(signal), -1, RATE).astype(float)
    available = np.isfinite(seconds)
    clean = np.where(available, seconds, 0)
    count = available.sum(-1)
    sum_ = clean.sum(-1)
    squares = (clean * clean).sum(-1)
    minimum = np.where(available, seconds, np.inf).min(-1)
    maximum = np.where(available, seconds, -np.inf).max(-1)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    mean = divide(sum_, count)
    span = np.where(count > 0, maximum - minimum, 0)
    diff = np.mean(np.abs(np.diff(clean, axis=-1)), axis=-1)
    last = np.arange(HISTORY - 1, seconds.shape[1])
    features, names = [], []

    def add(name, values, endpoints=last):
        chosen = values[endpoints]
        features.append(np.where(np.isfinite(chosen), chosen, 0).astype(np.float32))
        names.append(name)

    def past(values, width):
        return np.lib.stride_tricks.sliding_window_view(values, width)[last + 1 - width]

    if not use_cascade:
        for channel, label in enumerate(CHANNELS):
            for width in (5, 10, 30, 60):
                observed = roll_sum(count[channel], width)
                average = divide(roll_sum(sum_[channel], width), observed)
                var = variance(roll_sum(sum_[channel], width), roll_sum(squares[channel], width), observed)
                high = roll_extreme(np.where(count[channel] > 0, maximum[channel], -np.inf), width, True)
                low = roll_extreme(np.where(count[channel] > 0, minimum[channel], np.inf), width, False)
                add(f"{label}_{width}s_mean", average)
                add(f"{label}_{width}s_std", np.sqrt(var))
                add(f"{label}_{width}s_span", high - low)
                add(f"{label}_{width}s_activity", roll_sum(diff[channel], width) / width)
                add(f"{label}_{width}s_valid", observed / (RATE * width))
        for width in (10, 30):
            for left, right in ((0, 1), (0, 2), (1, 2)):
                a, b = mean[left], mean[right]
                av, bv = roll_sum(a, width) / width, roll_sum(b, width) / width
                numerator = roll_sum(a * b, width) / width - av * bv
                va = np.maximum(roll_sum(a ** 2, width) / width - av ** 2, 0)
                vb = np.maximum(roll_sum(b ** 2, width) / width - bv ** 2, 0)
                add(f"signed_sync_{CHANNELS[left]}_{CHANNELS[right]}_{width}s", divide(numerator, np.sqrt(va * vb)))
            for channel, label in enumerate(CHANNELS[:3]):
                history = np.lib.stride_tricks.sliding_window_view(clean[channel].ravel(), RATE * width)
                recent = history[(last + 1 - width) * RATE]
                spectrum = np.abs(np.fft.rfft((recent - recent.mean(-1, keepdims=True)) * np.hanning(RATE * width))) ** 2
                frequency = np.fft.rfftfreq(RATE * width, 1 / RATE)
                total = spectrum[:, 1:].sum(-1)
                for lower, upper in ((.1, .25), (.25, .5), (.5, 1.)):
                    band = spectrum[:, (frequency >= lower) & (frequency < upper)].sum(-1)
                    add(f"{label}_{width}s_power_{lower:g}_{upper:g}", divide(band, total), endpoints=slice(None))

    flow, oxygen = 0, len(signal) - 1
    widths = (10, 60) if use_cascade else (5, 10, 30)
    for width in widths:
        ref_ends = last - (30 if width == 60 else 10)
        flow_reference = roll_extreme(span[flow], 30, True)[ref_ends]
        oxygen_baseline = roll_extreme(np.where(count[oxygen] > 0, maximum[oxygen], -np.inf), 30, True)[ref_ends]
        oxygen_baseline = np.where(np.isfinite(oxygen_baseline), oxygen_baseline, mean[oxygen, last])
        recent_flow = past(span[flow], width)
        recent_oxygen = past(mean[oxygen], width)
        recent_oxygen_valid = past(count[oxygen], width) > 0
        observed = roll_sum(count[oxygen], width)
        oxygen_mean = divide(roll_sum(sum_[oxygen], width), observed)
        oxygen_var = variance(roll_sum(sum_[oxygen], width), roll_sum(squares[oxygen], width), observed)
        for threshold in (.3, .7):
            reduced = (flow_reference[:, None] > 0) & (recent_flow < (1 - threshold) * flow_reference[:, None])
            add(f"flow_dropped_{threshold:g}_{width}s", reduced.mean(axis=-1), endpoints=slice(None))
        add(f"flow_rel_activity_{width}s", divide(recent_flow.mean(-1), flow_reference), endpoints=slice(None))
        add(f"spo2_std_{width}s", np.sqrt(oxygen_var))
        add(f"spo2_trend_{width}s", mean[oxygen] - mean[oxygen][np.clip(np.arange(len(mean[oxygen])) - width + 1, 0, None)])
        add(f"spo2_relative_drop_{width}s", np.maximum(oxygen_baseline - oxygen_mean[last], 0), endpoints=slice(None))
        desat = (recent_oxygen < oxygen_baseline[:, None] * .98) & recent_oxygen_valid
        add(f"spo2_desaturation_{width}s", divide(desat.sum(-1), recent_oxygen_valid.sum(-1)), endpoints=slice(None))
        for threshold in (91, 92):
            below = np.count_nonzero((seconds[oxygen] < threshold) & available[oxygen], axis=-1)
            add(f"spo2_below_{threshold:g}_{width}s", divide(roll_sum(below, width), observed))
        add(f"spo2_valid_{width}s", observed / (RATE * width))
    return np.column_stack(features), tuple(names)


def training_selection(labels, seed, max_normal_ratio=4):
    normal = np.flatnonzero(labels == 0)
    events = np.flatnonzero(labels > 0)
    cap = min(len(normal), max_normal_ratio * len(events))
    included = np.random.default_rng(seed).choice(normal, cap, replace=False)
    return np.sort(np.r_[included, events])


def inverse_weights(labels):
    counts = np.bincount(labels)
    if np.any(counts == 0):
        raise ValueError("Fit sample does not cover the expected classes")
    return len(labels) / (len(counts) * counts[labels])


def make_forest(seed, threads, trees):
    return RandomForestClassifier(n_estimators=trees, max_depth=14, min_samples_leaf=5,
                                  max_features="sqrt", random_state=seed, n_jobs=threads)


@dataclass
class ClassicalFit:
    method: str
    feature_names: tuple
    scaler: StandardScaler
    primary: object
    secondary: object = None

    def column_mask(self, seconds):
        return np.asarray([name.endswith(f"_{seconds}s") for name in self.feature_names])

    def predict(self, features):
        x = self.scaler.transform(features)
        if self.method != "A06":
            return self.primary.predict_proba(x).astype(np.float32)
        coarse = self.primary.predict_proba(x[:, self.column_mask(60)])[:, 1]
        fine = self.secondary.predict_proba(x[:, self.column_mask(10)])[:, 1]
        ah = np.where(coarse >= .5, fine, 0.)
        return np.column_stack((1 - ah, ah)).astype(np.float32)


def fit_model(method, features, y, names, seed, threads, cascade_labels=None):
    scaler = StandardScaler().fit(features[y >= 0])
    features_scaled = scaler.transform(features)
    if method == "A06":
        model = ClassicalFit(method, names, scaler, make_forest(seed, threads, 10), make_forest(seed + 1, threads, 20))
        for duration, estimator, targets in zip((60, 10), (model.primary, model.secondary), cascade_labels):
            indices = training_selection(targets, seed)
            if sorted(set(targets[indices])) != [0, 1]:
                raise ValueError("Cascade fit is missing positive or negative labels")
            estimator.fit(features_scaled[indices][:, model.column_mask(duration)], targets[indices],
                          sample_weight=inverse_weights(targets[indices]))
        return model
    if method == "B04":
        primary = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.06, subsample=.8,
                                colsample_bytree=.8, eval_metric="mlogloss", n_jobs=threads,
                                tree_method="hist", random_state=seed)
    elif method == "B05":
        primary = make_forest(seed, threads, 120)
    else:
        raise ValueError(f"Unknown baseline {method}")
    indices = training_selection(y, seed)
    if sorted(set(y[indices])) != list(range(5)):
        raise ValueError("Fit sample lacks a five-class target")
    primary.fit(features_scaled[indices], y[indices], sample_weight=inverse_weights(y[indices]))
    return ClassicalFit(method, names, scaler, primary)
