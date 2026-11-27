import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef


def second_metrics(labels, probabilities):
    """Describe one mutually exclusive label space; retain absent-class support."""
    n_classes = probabilities.shape[-1]
    valid = labels >= 0
    y = labels[valid]
    p = probabilities[valid]
    if not len(y):
        raise ValueError("No scored seconds")
    pred = p.argmax(-1)
    classes = np.arange(n_classes)
    matrix = confusion_matrix(y, pred, labels=classes)
    hits = matrix.diagonal()
    support = matrix.sum(1)
    selected = matrix.sum(0)
    present = support > 0
    recall = hits / np.maximum(support, 1)
    precision = hits / np.maximum(selected, 1)
    f1 = 2 * hits / np.maximum(support + selected, 1)
    ap = [float(average_precision_score(y == i, p[:, i])) if 0 < support[i] < len(y) else None
          for i in classes]
    defined_ap = [value for value in ap if value is not None]
    bin_ids = (p.max(-1) * 15).astype(int).clip(0, 14)
    ece = 0.
    for bin_id in np.unique(bin_ids):
        use = bin_ids == bin_id
        ece += use.mean() * abs((y[use] == pred[use]).mean() - p[use].max(-1).mean())
    return dict(
        scored_seconds=len(y), accuracy=float(hits.sum() / len(y)),
        confusion_matrix=matrix.tolist(), balanced_accuracy=float(recall[present].mean()),
        macro_f1=float(f1[present].mean()), mcc=float(matthews_corrcoef(y, pred)),
        macro_auprc=float(np.mean(defined_ap)) if defined_ap else None, ece=float(ece),
        precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(), average_precision=ap,
        support=support.tolist(),
    )
