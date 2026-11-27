"""Project record-relative event annotations onto one-second sleep labels."""

from dataclasses import dataclass

import numpy as np

from .ucddb_mapping import IGNORE, map_event


@dataclass(frozen=True)
class Timeline:
    labels: np.ndarray
    stage_by_second: np.ndarray
    coverage: np.ndarray
    event_index: np.ndarray
    event_start_distance: np.ndarray
    event_end_distance: np.ndarray


def rasterize(duration: int, stages: np.ndarray, events) -> Timeline:
    stage = np.full(duration, IGNORE, dtype=np.int8)
    stage_seconds = np.repeat(stages, 30)[:duration]
    stage[:len(stage_seconds)] = stage_seconds
    labels = np.zeros(duration, dtype=np.int8)
    coverage = np.zeros(duration, dtype=np.int32)
    index = np.full(duration, IGNORE, dtype=np.int32)
    start_distance = np.full(duration, np.nan)
    end_distance = np.full(duration, np.nan)
    for idx, row in enumerate(events.itertuples(index=False)):
        start, end = float(row.start_sec), float(row.end_sec)
        if not 0 <= start < end <= duration:
            raise ValueError(f"Invalid event interval {start}, {end}")
        first, stop = np.ceil(np.array([start, end]) - .5).astype(int)
        sl = slice(first, stop)
        label = map_event(row.original_label)
        periodic = getattr(row, "periodic_flag", "")
        if isinstance(periodic, str) and periodic.strip():
            label = IGNORE
        midpoints = np.arange(first, stop) + .5
        coverage[sl] += 1
        labels[sl] = label
        index[sl] = idx
        start_distance[sl] = midpoints - start
        end_distance[sl] = end - midpoints
    labels[(coverage > 1) | ~np.isin(stage, [1, 2, 3, 4, 5])] = IGNORE
    index[labels == IGNORE] = IGNORE
    start_distance[labels == IGNORE] = np.nan
    end_distance[labels == IGNORE] = np.nan
    return Timeline(labels, stage, coverage, index, start_distance, end_distance)
