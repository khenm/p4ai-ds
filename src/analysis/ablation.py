"""Stage-1 feature ablation for the LightGBM Stage-2 classifier.

Each group of columns (image embeddings, per-head logit blocks, PhotoAmt) is
zeroed out in turn, then Stage-2 is re-run.  The resulting accuracy and QWK
drops show which inputs the final AdoptionSpeed model relies on most.
"""

from itertools import combinations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


# Ordered list matches the loop in train_lgb.py
_STAGE1_ORDER = [
    "Type", "FurLength", "MaturitySize", "Breed1",
    "Health", "Vaccinated", "Dewormed", "Sterilized", "Gender", "Color1",
]


def compute_feature_slices(
    img_dim: int,
    stage1_clfs: dict,
    has_photo_amt: bool = True,
) -> dict[str, tuple[int, int]]:
    """Return start/end column indices for each feature group in X_concat.

    Args:
        img_dim: number of image-embedding dimensions (512 for ResNet18).
        stage1_clfs: ordered dict mapping head name → fitted LGBMClassifier,
                     in the same order they were appended to X_concat.
        has_photo_amt: whether the last column is PhotoAmt.

    Returns:
        dict of group_name → (start, end) half-open slice.
    """
    slices: dict[str, tuple[int, int]] = {"image": (0, img_dim)}
    cursor = img_dim
    for name in _STAGE1_ORDER:
        if name not in stage1_clfs:
            continue
        n = len(stage1_clfs[name].classes_)
        slices[name] = (cursor, cursor + n)
        cursor += n
    if has_photo_amt:
        slices["PhotoAmt"] = (cursor, cursor + 1)
    return slices


def run_ablation(
    clf,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_slices: dict[str, tuple[int, int]],
) -> dict:
    """Ablate each feature group and report accuracy / QWK impact.

    Args:
        clf: fitted Stage-2 LGBMClassifier.
        X_val: validation feature matrix (n_samples, n_features).
        y_val: true AdoptionSpeed labels (n_samples,).
        feature_slices: output of :func:`compute_feature_slices`.

    Returns:
        dict with keys:
          - ``baseline``: dict(accuracy, qwk) on unmodified X_val.
          - ``ablations``: list of dicts, one per group, each with
            group, accuracy, qwk, delta_accuracy, delta_qwk.
    """
    baseline_pred = clf.predict(X_val)
    baseline_acc = accuracy_score(y_val, baseline_pred)
    baseline_qwk = cohen_kappa_score(y_val, baseline_pred, weights="quadratic")

    ablations = []
    for group, (start, end) in feature_slices.items():
        X_ablated = X_val.copy()
        X_ablated[:, start:end] = 0.0
        pred = clf.predict(X_ablated)
        acc = accuracy_score(y_val, pred)
        qwk = cohen_kappa_score(y_val, pred, weights="quadratic")
        ablations.append({
            "group": group,
            "accuracy": round(acc, 4),
            "qwk": round(qwk, 4),
            "delta_accuracy": round(acc - baseline_acc, 4),
            "delta_qwk": round(qwk - baseline_qwk, 4),
        })

    ablations.sort(key=lambda x: x["delta_qwk"])

    return {
        "baseline": {
            "accuracy": round(baseline_acc, 4),
            "qwk": round(baseline_qwk, 4),
        },
        "ablations": ablations,
    }


def run_ablation_combinations(
    clf,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_slices: dict[str, tuple[int, int]],
    max_combo_size: int | None = None,
) -> dict:
    """Evaluate every non-empty subset of feature groups (power-set ablation).

    With N groups this runs 2^N − 1 forward passes through the classifier.
    Use ``max_combo_size`` to cap at combinations of size k or fewer when N
    is large and full enumeration is too slow.

    Args:
        clf: fitted Stage-2 LGBMClassifier.
        X_val: validation feature matrix (n_samples, n_features).
        y_val: true AdoptionSpeed labels (n_samples,).
        feature_slices: output of :func:`compute_feature_slices`.
        max_combo_size: if set, only test subsets of this size or smaller.
                        ``None`` tests all 2^N − 1 subsets.

    Returns:
        dict with keys:
          - ``baseline``: dict(accuracy, qwk) on unmodified X_val.
          - ``n_combinations``: total number of subsets evaluated.
          - ``combinations``: list of dicts sorted by delta_qwk (ascending).
          - ``worst``: top-5 combinations with the largest QWK drop.
          - ``best``: top-5 combinations with the smallest QWK drop (most
            redundant groups — removing them barely hurts).
    """
    baseline_pred = clf.predict(X_val)
    baseline_acc = accuracy_score(y_val, baseline_pred)
    baseline_qwk = cohen_kappa_score(y_val, baseline_pred, weights="quadratic")

    groups = list(feature_slices.keys())
    cap = max_combo_size if max_combo_size is not None else len(groups)

    results = []
    for size in range(1, cap + 1):
        for combo in combinations(groups, size):
            X_ab = X_val.copy()
            for g in combo:
                s, e = feature_slices[g]
                X_ab[:, s:e] = 0.0
            pred = clf.predict(X_ab)
            acc = accuracy_score(y_val, pred)
            qwk = cohen_kappa_score(y_val, pred, weights="quadratic")
            results.append({
                "ablated": list(combo),
                "n_ablated": size,
                "accuracy": round(acc, 4),
                "qwk": round(qwk, 4),
                "delta_accuracy": round(acc - baseline_acc, 4),
                "delta_qwk": round(qwk - baseline_qwk, 4),
            })

    results.sort(key=lambda x: x["delta_qwk"])

    return {
        "baseline": {
            "accuracy": round(baseline_acc, 4),
            "qwk": round(baseline_qwk, 4),
        },
        "n_combinations": len(results),
        "combinations": results,
        "worst": results[:5],
        "best": results[-5:],
    }
