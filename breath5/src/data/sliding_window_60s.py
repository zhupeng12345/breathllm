"""Sixty-second causal inputs and five one-second supervision targets."""

import numpy as np

HISTORY_SECONDS = 60
TAIL_SECONDS = 5
SAMPLE_RATE = 8


def window_ends(duration: int) -> np.ndarray:
    return np.arange(HISTORY_SECONDS, duration + 1, dtype=int)


def tail_indices(ends: np.ndarray) -> np.ndarray:
    return np.asarray(ends)[:, None] + np.arange(-TAIL_SECONDS, 0)


def tail_labels(labels: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return labels[tail_indices(ends)]


def training_weights(labels: np.ndarray, ends: np.ndarray) -> np.ndarray:
    indices = tail_indices(ends)
    counts = np.bincount(indices.ravel(), minlength=len(labels))
    return np.where(labels[indices] >= 0, 1 / counts[indices], 0.)


def waveform_window(signal: np.ndarray, end: int) -> np.ndarray:
    start = (end - HISTORY_SECONDS) * SAMPLE_RATE
    stop = end * SAMPLE_RATE
    if start < 0 or stop > signal.shape[-1]:
        raise ValueError("Window outside available waveform")
    return signal[:, start:stop]


def released_prediction(probabilities: np.ndarray) -> np.ndarray:
    return probabilities[..., -1, :]
