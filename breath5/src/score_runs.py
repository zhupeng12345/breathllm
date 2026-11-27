"""Score sealed per-night probabilities with one validation-only alarm protocol."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.experiment_data import ROOT, TARGETS
from src.metrics import second_metrics


def runs(labels, gap=0, min_positive=1):
    previous = np.r_[0, labels[:-1]]
    begins = np.flatnonzero((labels > 0) & (labels != previous))
    ends = np.r_[np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    stops = ends[np.searchsorted(ends, begins, side="right")]
    result = []
    for a, b in zip(begins, stops):
        cls = int(labels[a])
        if result and cls == result[-1][2] and a - result[-1][1] <= gap and not labels[result[-1][1]:a].any():
            result[-1][1] = b
            result[-1][3] += b-a
        else:
            result.append([a, b, cls, b-a])
    return np.asarray([r[:3] for r in result if r[3] >= min_positive], dtype=int).reshape(-1, 3)


def totals(y, p, threshold, *, details=False, detect_first=False, gap=0, min_run=3):
    valid = y >= 0
    positive = valid & (p[:, 1:].sum(1) >= threshold)
    if detect_first:
        alarms = runs(positive.astype(int), gap=gap, min_positive=min_run)
        cumulative = np.vstack((np.zeros(p.shape[1]-1), np.cumsum(p[:, 1:] * positive[:, None], axis=0)))
        alarms[:, 2] = (cumulative[alarms[:, 1]] - cumulative[alarms[:, 0]]).argmax(1) + 1
    else:
        pred = np.where(positive, p[:, 1:].argmax(1) + 1, 0)
        alarms = runs(pred, gap=gap, min_positive=min_run)
    truth = runs(np.where(valid, y, 0))
    overlap = np.maximum(0, np.minimum(alarms[:, None, 1], truth[None, :, 1])
                         - np.maximum(alarms[:, None, 0], truth[None, :, 0]))
    enough = overlap >= np.minimum(5, (truth[:, 1] - truth[:, 0]))[None, :]
    row, col = linear_sum_assignment(np.where(enough, 1 + overlap / (1 + overlap.sum()), 0), maximize=True)
    keep = enough[row, col]
    row, col = row[keep], col[keep]
    typed = alarms[row, 2] == truth[col, 2]
    counts = dict(seconds=int(valid.sum()), predicted=len(alarms), truth=len(truth),
                  detected=len(row), typed=int(typed.sum()))
    if details:
        matches = np.full((len(truth), 3), np.nan)
        matches[col, 0] = alarms[row, 2]
        matches[col, 1:] = alarms[row, :2] - truth[col, :2]
        return counts, np.column_stack((truth, matches))
    return counts


def event_metrics(counts):
    c = pd.DataFrame(counts).sum()
    truth, predicted, detected, typed = (c[k] for k in ("truth", "predicted", "detected", "typed"))
    return dict(events=int(truth), alarms=int(predicted), detected=int(detected), correct_subtype=int(typed),
                event_f1=2*detected/max(truth+predicted, 1), typed_event_f1=2*typed/max(truth+predicted, 1),
                sensitivity=detected/max(truth, 1), event_precision=detected/max(predicted, 1),
                false_alarms_per_sleep_hour=(predicted-detected)*3600/max(c.seconds, 1))


def project(y, p, task):
    if task == "five":
        return y, p
    n, o, c, m, h = p.T
    if task == "yu":
        return np.array([0, 2, 3, 3, 1, -1])[y], np.stack((n, h, o, c+m), 1)
    return np.where(y < 0, -1, (y > 0).astype(int)), np.stack((n, o+c+m+h), 1)


def paired_bootstrap(rows, seed=4121):
    rng = np.random.default_rng(seed)
    values = np.array([[item[k] for k in ("detected", "typed", "truth", "predicted")] for item in rows])
    sampled = values[rng.integers(len(values), size=(2000, len(values)))].sum(1)
    denom = np.maximum(sampled[:, 2] + sampled[:, 3], 1)
    interval = np.quantile(2*sampled[:, :2]/denom[:, None], [.025, .975], axis=0)
    return dict(event_ci_low=interval[0, 0], event_ci_high=interval[1, 0],
                typed_ci_low=interval[0, 1], typed_ci_high=interval[1, 1])


def probability_files(folder, expected, reference, method):
    paths = list(folder.glob("*.npz"))
    if {p.stem for p in paths} != set(expected):
        return None
    result = {}
    for path in sorted(paths):
        y, seconds = reference[path.stem]
        with np.load(path) as saved:
            np.testing.assert_array_equal(seconds, saved["second"])
            p = saved["probabilities"]
        if not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1, atol=1e-4):
            raise ValueError(f"Invalid forecast {path}")
        if method == "A04":
            y = np.array([0, 1, -1, -1, -1, -1])[y]
        elif p.shape[1] == 2:
            y = np.where(y < 0, -1, (y > 0).astype(int))
        elif p.shape[1] == 4:
            y = np.array([0, 2, 3, 3, 1, -1])[y]
        result[path.stem] = (y, p)
    return result


def report():
    install = ROOT / "experiments"
    split = pd.read_csv(ROOT / "audit/split_manifest.csv")
    reference = {
        person: (x[TARGETS[-1]].to_numpy(), x.input_end_sec.to_numpy()-1)
        for person, x in pd.read_csv(ROOT / "audit/window_manifest.csv").groupby("record_id")
    }
    plan = pd.read_csv(install / "comparison_plan.csv").set_index("experiment_id")
    pooled = {}
    subjects, folds, events = [], [], []
    folders = (path for source in ("classical", "deep", "innovation")
               for path in (install / source).glob("*/fold_*") if path.is_dir())
    for folder in sorted(folders):
        method = folder.parent.name
        if method not in plan.index:
            continue
        fold = int(folder.name.split("_")[-1])
        roles = split.loc[split.fold.eq(fold)].groupby("control_role").subject_id.agg(list)
        score = probability_files(folder / "predictions", roles["score"], reference, method)
        validation = probability_files(folder / "validation", roles["select"], reference, method)
        if score is None or validation is None:
            continue
        shape = next(iter(score.values()))[1].shape[1]
        tasks = ["five", "yu", "binary"] if shape == 5 else ["yu"] if shape == 4 else [
            "OSA_detection" if method == "A04" else "binary"]
        for task in tasks:
            v = list(validation.values())
            s = score
            if shape == 5:
                v = [project(y, p, task) for y, p in v]
                s = {person: project(y, p, task) for person, (y, p) in score.items()}
            candidates = []
            for threshold in (.35, .50, .65, .80, .90, .95):
                em = event_metrics([totals(y, p, threshold) for y, p in v])
                candidates.append((em["event_f1"], em["typed_event_f1"], -em["false_alarms_per_sleep_hour"], threshold))
            threshold = max(candidates)[-1]
            counts = []
            for person, (y, p) in s.items():
                count, match = totals(y, p, threshold, details=True)
                counts.append(count)
                subjects.append(dict(method=method, task=task, fold=fold, record_id=person, threshold=threshold,
                                     **second_metrics(y, p), **event_metrics([count])))
                for a, b, cls, detected, onset, offset in match:
                    events.append(dict(method=method, task=task, fold=fold, record_id=person, start_sec=int(a+59),
                                       end_sec=int(b+59), reference=int(cls), detected_as=detected,
                                       onset_error_sec=onset, offset_error_sec=offset))
            y = np.concatenate([pair[0] for pair in s.values()])
            p = np.concatenate([pair[1] for pair in s.values()])
            folds.append(dict(method=method, task=task, fold=fold, threshold=threshold,
                              **second_metrics(y, p), **event_metrics(counts)))
            pooled.setdefault((method, task), []).append((y, p, counts))
    output = install / "summary"
    output.mkdir(exist_ok=True)
    table = []
    for (method, task), parts in pooled.items():
        y = np.concatenate([r[0] for r in parts])
        p = np.concatenate([r[1] for r in parts])
        counts = [c for row in parts for c in row[2]]
        table.append(dict(method=method, name=plan.method.get(method, method), task=task,
                          track=plan.track.get(method, "B"), folds=len(parts), subjects=len(counts),
                          **second_metrics(y, p), **event_metrics(counts), **paired_bootstrap(counts)))
    pd.DataFrame(table).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(subjects).to_csv(output / "subject_metrics.csv", index=False)
    pd.DataFrame(events).to_csv(output / "event_matches.csv", index=False)
    print(f"{len(table)} comparison rows, {len(folds)} fold rows")


if __name__ == "__main__":
    report()
