import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def build_clf_section(targets, preds, class_names=None):
    """Return dict with sklearn classification_report string and confusion matrix."""
    return {
        "classification_report": classification_report(
            targets, preds, target_names=class_names, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
    }


ADOPTION_SPEED_NAMES = [
    "Same day",       # 0
    "1–7 days",       # 1
    "8–30 days",      # 2
    "31–90 days",     # 3
    "No adoption",    # 4
]


def _to_int_list(arr):
    """Flatten any array-like to a plain Python list of ints."""
    return np.asarray(arr).ravel().astype(int).tolist()


def build_adoption_speed_section(targets, preds):
    """Classification report + confusion matrix for AdoptionSpeed (5 classes)."""
    targets = _to_int_list(targets)
    preds   = _to_int_list(preds)
    labels  = sorted(set(targets + preds))
    class_names = [ADOPTION_SPEED_NAMES[i] for i in labels]
    return build_clf_section(targets, preds, class_names=class_names)


def build_breed_section(targets, preds):
    """Classification report + confusion matrix for Breed1 (up to 308 classes).

    Class names are omitted because breed IDs have no human-readable mapping here;
    the report uses numeric breed codes instead.
    """
    return build_clf_section(_to_int_list(targets), _to_int_list(preds), class_names=None)


def save_report(config_path, report_data, report_dir="results/reports"):
    """Write report_data as JSON to results/reports/<config_stem>.json."""
    os.makedirs(report_dir, exist_ok=True)
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    out_path = os.path.join(report_dir, f"{config_stem}.json")
    report_data.update({
        "config": config_stem,
        "timestamp": datetime.now().isoformat(),
    })
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    return out_path
