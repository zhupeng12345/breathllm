"""T03 来源、预处理和整夜评分的共用校验函数。"""

import hashlib
import sqlite3
from contextlib import closing, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import B07基座改造实验 as b07
from src import experiment_data

track = b07.track


VERSION = "cce921d90c27ab2a-F02-R07"
FOLD, SEED = 1, 4121


T03 = ((.5, 0., .5), 0., 0., "original", "new", "random")


KD = (T03[0], .05, *T03[2:])


KD_TEMPERATURE = 2.


ATOL = 1e-6


PERTURBATIONS = ("noise", "gain", *track.PSG, "missing")


def readonly(path):
    return closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True))


def digest(*items):
    sha = hashlib.sha256()
    for item in items:
        sha.update(item)
    return sha.hexdigest()


def read_native_prefix(path, seconds):
    with experiment_data.pyedflib.EdfReader(str(path)) as reader:
        indices = {name.casefold(): i for i, name in enumerate(reader.getSignalLabels())}
        rates, samples = [], []
        for channel in track.PSG:
            index = indices[channel.casefold()]
            rate = int(reader.getSampleFrequency(index))
            values = reader.readSignal(index, n=seconds * rate)
            if len(values) != seconds * rate:
                raise ValueError(f"{path.name} has an incomplete native {channel} prefix")
            rates.append(rate)
            samples.append(values)
    return rates, samples


def preprocess_prefix(rates, samples):
    # Exercise the archived waveform function with supplied native EDF samples.
    reader = SimpleNamespace(getSignalLabels=lambda: track.PSG, getSampleFrequency=rates.__getitem__,
                             readSignal=lambda index: samples[index].copy())
    proxy = SimpleNamespace(EdfReader=lambda _path: nullcontext(reader))
    with patch.object(experiment_data, "pyedflib", proxy):
        return experiment_data.waveform("native prefix", track.PSG)


def perturb_native(samples, rates, anchor, mode, rng):
    changed = [signal.copy() for signal in samples]
    for channel, signal, rate in zip(track.PSG, changed, rates):
        future = signal[anchor * rate:]
        if mode == "missing":
            future[:] = np.nan
        elif mode == "gain":
            future *= 1.7
        elif mode == "noise" or mode == channel:
            future += rng.normal(0., max(float(np.std(signal[:anchor * rate])), 1.), len(future))
    return changed


def standardize(wave, mean, scale):
    return np.nan_to_num((wave - mean[:, None]) / scale[:, None]).astype(np.float32)


def check_clock(seconds, start):
    if seconds.ndim != 1 or seconds.dtype.kind not in "iu" or not len(seconds):
        raise ValueError("Expected a nonempty integer prediction clock")
    np.testing.assert_array_equal(seconds, np.arange(start, start + len(seconds)))


def check_predictions(p, seconds, expected):
    check_clock(seconds, int(expected[0]))
    np.testing.assert_array_equal(seconds, expected)
    if (p.shape != (len(seconds), 5) or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(1), 1, atol=2e-5)):
        raise ValueError("Invalid five-class probabilities")


def reference_labels(roster):
    roles = roster[FOLD]
    development = set(roles["fit"] + roles["select"])
    validation = [person for partition in roster.values() for person in partition["select"]]
    if len(development) != 20 or sorted(validation) != sorted(development):
        raise ValueError("Expected 20 distinct development subjects across the three validation folds")
    labels = b07.labels_for(roles)
    if labels.keys() != development:
        raise ValueError("Missing development subjects in the label manifest")
    for targets, seconds in labels.values():
        check_clock(seconds, 59)
        if targets.shape != (len(seconds), 5) or not np.isin(targets, [-1, 0, 1, 2, 3, 4]).all():
            raise ValueError("Invalid label matrix")
        np.testing.assert_array_equal(targets[:-1, 1:], targets[1:, :-1])
    return labels


def check_matches(y, positive, match):
    # The scorer stores half-open reference spans and signed alarm-minus-reference offsets.
    np.testing.assert_array_equal(match[:, :3], b07.runs(np.maximum(y, 0)))
    detected = np.isfinite(match[:, 3])
    if not np.isnan(match[~detected, 3:]).all():
        raise ValueError("Unmatched reference has alarm details")
    details = match[detected, 3:]
    if (not np.isfinite(details).all() or not np.equal(details, np.trunc(details)).all()
            or not np.isin(details[:, 0], [1, 2, 3, 4]).all()):
        raise ValueError("Invalid matched class or offset")
    reference = match[detected, :2].astype(int)
    alarm = reference + details[:, 1:].astype(int)
    candidates = b07.runs(positive.astype(int), gap=b07.OPTIONS["gap"], min_positive=b07.OPTIONS["min_run"])
    indices = np.searchsorted(candidates[:, 0], alarm[:, 0])
    if (indices >= len(candidates)).any() or len(np.unique(indices)) != len(indices):
        raise ValueError("Matched alarms must be valid and one-to-one")
    np.testing.assert_array_equal(alarm, candidates[indices, :2])
    overlap = np.minimum(alarm[:, 1], reference[:, 1]) - np.maximum(alarm[:, 0], reference[:, 0])
    if (overlap < np.minimum(5, reference[:, 1] - reference[:, 0])).any():
        raise ValueError("Matched alarm has insufficient overlap")


def fixed_labels(match, base_positive, p):
    detected = np.isfinite(match[:, 3])
    start, stop = (match[detected, :2] + match[detected, 4:6]).astype(int).T
    total = np.vstack((np.zeros(4), (p[:, 1:] * base_positive[:, None]).cumsum(axis=0)))
    result = np.zeros(len(match), dtype=int)
    result[detected] = (total[stop] - total[start]).argmax(1) + 1
    return result
