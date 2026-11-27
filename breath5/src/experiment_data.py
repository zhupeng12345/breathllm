"""Audited record intervals and fit-local waveform scaling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, sosfilt

from breathllm_paths import DATA, MANIFEST, PROJECT

ROOT = Path(__file__).absolute().parents[1]
PSG = ("Flow", "ribcage", "abdo", "SpO2")
TARGETS = [f"target_{j}_label" for j in range(5)]


def waveform(path, channels):
    with pyedflib.EdfReader(str(path)) as edf:
        names = {name.casefold(): i for i, name in enumerate(edf.getSignalLabels())}
        result = []
        for name in channels:
            idx = names[name.casefold()]
            rate = int(edf.getSampleFrequency(idx))
            x = edf.readSignal(idx)
            if rate != 8:
                if rate % 8:
                    raise ValueError(f"Unsupported sample rate {rate}")
                x = sosfilt(butter(4, 3., fs=rate, output="sos"), x)[rate//8-1::rate//8]
            if name == "SpO2":
                x[(x < 50) | (x > 100)] = np.nan
            result.append(x)
    return np.stack(result).astype(np.float32)


class FoldData:
    def __init__(self, fold, channels=PSG, project=PROJECT, roles=None):
        if roles is None:
            split = pd.read_csv(ROOT / "audit/split_manifest.csv").query("fold == @fold")
            roles = {role: sorted(split.loc[split.control_role == role, "subject_id"])
                     for role in ("fit", "select", "score", "reserved")}
        self.roles = roles
        assigned = [person for group in roles.values() for person in group]
        if len(set(assigned)) != len(assigned):
            raise ValueError("Overlapping subject roles")
        if not roles["fit"] or not (roles["select"] or roles["score"]):
            raise ValueError(f"Missing partition in fold {fold}")
        self.channels = channels
        self.signals = {
            person: waveform(DATA / f"{person}.rec", channels)
            for role in ("fit", "select", "score") for person in self.roles[role]
        }
        fit = [self.signals[p].astype(np.float64) for p in self.roles["fit"]]
        counts = sum(np.isfinite(x).sum(1) for x in fit)
        self.mean = sum(np.nansum(x, axis=1) for x in fit) / counts
        self.scale = np.sqrt(sum(np.nansum((x - self.mean[:, None])**2, axis=1) for x in fit) / counts)
        if not (np.isfinite(self.mean).all() and np.isfinite(self.scale).all() and (self.scale > 0).all()):
            raise ValueError("Degenerate fit-side scaling")
        for person, x in self.signals.items():
            self.signals[person] = np.nan_to_num((x - self.mean[:, None]) / self.scale[:, None]).astype(np.float32)
        self.ends = {p: np.arange(60, x.shape[1] // 8 + 1) for p, x in self.signals.items()}
        manifest = pd.read_csv(MANIFEST)
        self.labels = {}
        for p in self.roles["fit"] + self.roles["select"]:
            rows = manifest.loc[manifest.record_id.eq(p)]
            np.testing.assert_array_equal(self.ends[p], rows.input_end_sec.to_numpy())
            self.labels[p] = rows[TARGETS].to_numpy(dtype=np.int8)

    def windows(self, person, indices):
        starts = (self.ends[person][indices] - 60) * 8
        return self.signals[person][:, starts[:, None] + np.arange(480)].transpose(1, 0, 2).copy()

    def fit_pool(self, seed, limit=1100):
        """Balance sample count by class, then by subject within each class."""
        rng = np.random.default_rng(seed)
        chosen = {}
        for cls in range(5):
            available = {p: np.flatnonzero(self.labels[p][:, -1] == cls) for p in self.roles["fit"]}
            present = [p for p, indices in available.items() if len(indices)]
            for p in present:
                ids = available[p]
                sampled = rng.choice(ids, min(len(ids), int(np.ceil(limit / len(present)))), replace=False)
                chosen.setdefault(p, []).extend(sampled)
        return {p: np.unique(indices) for p, indices in chosen.items()}
