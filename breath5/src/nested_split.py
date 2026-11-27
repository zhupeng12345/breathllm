"""Create a deterministic, class-covered, subject-level nested manifest."""

import hashlib
import json

import numpy as np
import pandas as pd

from src.experiment_data import ROOT


OUTPUT = ROOT / "experiments/paper_eval"
SPLIT = OUTPUT / "nested_split.csv"
AUDIT = ROOT / "audit/subject_class_coverage.csv"
SEED = 4121


def partition(counts, n_splits, seed):
    values = counts.to_numpy()
    target = values.sum(0) / n_splits
    coverage = (values > 0).sum(0) / n_splits
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    sizes = np.cumsum([len(a) for a in np.array_split(np.arange(len(values)), n_splits)])[:-1]
    for _ in range(12000):
        subsets = np.split(rng.permutation(len(values)), sizes)
        seconds = np.stack([values[idx].sum(0) for idx in subsets])
        n_people = np.stack([(values[idx] > 0).sum(0) for idx in subsets])
        if (n_people == 0).any():
            continue
        loss = ((seconds / target - 1) ** 2).sum() + .2 * ((n_people / coverage - 1) ** 2).sum()
        if loss < best[0]:
            best = (loss, subsets)
    if best[1] is None:
        raise ValueError("No grouping with all five classes in each fold")
    return [sorted(counts.index[idx]) for idx in best[1]]


def build():
    coverage = pd.read_csv(AUDIT)
    counts = coverage.pivot(index="subject_id", columns="class_id", values="window_target_seconds")
    outcomes = []
    people = set(counts.index)
    for fold, outer in enumerate(partition(counts, 5, SEED), 1):
        fit = sorted(people - set(outer))
        for inner, validation in enumerate(partition(counts.loc[fit], 3, SEED + fold), 1):
            remaining = sorted(set(fit) - set(validation))
            for role, group in (("fit", remaining), ("select", validation), ("reserved", outer)):
                outcomes.extend(dict(outer=fold, inner=inner, role=role, subject_id=p) for p in group)
        for role, group in (("fit", fit), ("score", outer)):
            outcomes.extend(dict(outer=fold, inner=0, role=role, subject_id=p) for p in group)

    frame = pd.DataFrame(outcomes)
    for outer, branch in frame.groupby("outer"):
        scores = branch.loc[branch.role.eq("score"), "subject_id"].tolist()
        selects = branch.loc[branch.role.eq("select"), "subject_id"].tolist()
        if sorted(scores + selects) != sorted(people):
            raise ValueError(f"Outer fold {outer} inner holdouts are not an exact partition")
    if sorted(frame.loc[frame.role.eq("score"), "subject_id"]) != sorted(people):
        raise ValueError("Outer test subjects do not appear exactly once")

    metrics = []
    for (outer, inner, role), rows in frame.groupby(["outer", "inner", "role"]):
        sub = counts.loc[rows.subject_id]
        for cls in counts.columns:
            metrics.append(dict(outer=outer, inner=inner, role=role, class_id=cls,
                                subjects=len(sub), subjects_with_class=int(sub[cls].gt(0).sum()),
                                supervised_seconds=int(sub[cls].sum())))
    totals = pd.DataFrame(metrics)
    if (totals.subjects_with_class == 0).any():
        raise ValueError("One partition is missing a target class")
    OUTPUT.mkdir(exist_ok=True)
    if SPLIT.exists():
        pd.testing.assert_frame_equal(frame, pd.read_csv(SPLIT))
    else:
        frame.to_csv(SPLIT, index=False)
    totals.to_csv(OUTPUT / "class_coverage.csv", index=False)
    (OUTPUT / "split_receipt.json").write_text(json.dumps({
        "seed": SEED, "outer_folds": 5, "inner_folds": 3,
        "coverage_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print(totals.query("inner == 0 and role == 'score'").pivot(
        index="outer", columns="class_id", values="subjects_with_class").to_string())


def roles_for(frame, outer, inner):
    part = frame.loc[(frame.outer == outer) & (frame.inner == inner)]
    return {role: sorted(part.loc[part.role.eq(role), "subject_id"])
            for role in ("fit", "select", "score", "reserved")}


def fold_folder(outer, inner):
    return OUTPUT / "runs" / f"outer_{outer}" / ("refit" if inner == 0 else f"inner_{inner}")


if __name__ == "__main__":
    build()
