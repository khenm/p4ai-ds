"""SHAP-based feature importance for the LightGBM Stage-2 classifier.

Uses TreeExplainer to decompose each AdoptionSpeed prediction into per-feature
contributions, then aggregates by logical group (image embeddings, each Stage-1
head's logit block, PhotoAmt).
"""

import numpy as np
import shap


# Ordered list matches the loop in train_lgb.py
_STAGE1_ORDER = [
    "Type", "FurLength", "MaturitySize", "Breed1",
    "Health", "Vaccinated", "Dewormed", "Sterilized", "Gender", "Color1",
]


def build_feature_names(
    img_dim: int,
    stage1_clfs: dict,
    has_photo_amt: bool = True,
) -> list[str]:
    """Return one name per column of X_concat, matching the layout in train_lgb.py.

    Args:
        img_dim: number of image-embedding dimensions (512 for ResNet18).
        stage1_clfs: ordered dict mapping head name → fitted LGBMClassifier.
        has_photo_amt: whether the last column is PhotoAmt.

    Returns:
        List of strings such as ``["image[0]", ..., "Type[0]", ..., "PhotoAmt"]``.
    """
    names = [f"image[{i}]" for i in range(img_dim)]
    for head in _STAGE1_ORDER:
        if head not in stage1_clfs:
            continue
        n = len(stage1_clfs[head].classes_)
        names += [f"{head}[{i}]" for i in range(n)]
    if has_photo_amt:
        names.append("PhotoAmt")
    return names


def run_shap(
    clf,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 500,
) -> dict:
    """Compute SHAP values and aggregate importance by feature group.

    Args:
        clf: fitted Stage-2 LGBMClassifier.
        X: feature matrix to explain (n_samples, n_features).
        feature_names: list from :func:`build_feature_names`.  When provided,
                       results are grouped by the prefix before ``[``.
        max_samples: cap on rows to keep runtime manageable; random sample is taken.

    Returns:
        dict with keys:
          - ``group_importance``: dict mapping group name → mean |SHAP| across
            all classes and samples, sorted descending.
          - ``top_features``: list of dicts (name, importance) for the 20 most
            important individual features.
          - ``n_samples_used``: actual row count passed to TreeExplainer.
    """
    rng = np.random.default_rng(42)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X_explain)

    # raw is (n_samples, n_features, n_classes) or list of (n_samples, n_features)
    if isinstance(raw, list):
        sv = np.stack(raw, axis=-1)        # (n_samples, n_features, n_classes)
    else:
        sv = raw if raw.ndim == 3 else raw[..., np.newaxis]

    mean_abs = np.abs(sv).mean(axis=(0, 2))    # (n_features,)

    # Per-feature top-20
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_features = [
        {
            "feature": feature_names[i] if feature_names else f"feature_{i}",
            "importance": round(float(mean_abs[i]), 6),
        }
        for i in top_idx
    ]

    # Group-level importance
    group_importance: dict[str, float] = {}
    if feature_names:
        for i, name in enumerate(feature_names):
            group = name.split("[")[0]
            group_importance[group] = group_importance.get(group, 0.0) + float(mean_abs[i])
        group_importance = dict(
            sorted(group_importance.items(), key=lambda kv: kv[1], reverse=True)
        )
    else:
        group_importance["all"] = float(mean_abs.sum())

    return {
        "group_importance": {k: round(v, 6) for k, v in group_importance.items()},
        "top_features": top_features,
        "n_samples_used": int(X_explain.shape[0]),
    }
