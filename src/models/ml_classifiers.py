"""Factory for interchangeable sklearn-compatible ML classifiers.

Supported names
---------------
lightgbm      LGBMClassifier      (fast gradient boosting, GPU via OpenCL)
xgboost       XGBClassifier       (gradient boosting, GPU via CUDA/hist)
catboost      CatBoostClassifier  (gradient boosting, GPU native)
decision_tree DecisionTreeClassifier
svm           SVC with probability=True  (slow on large feature sets)
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SUPPORTED = ("lightgbm", "xgboost", "catboost", "decision_tree", "svm")


class _XGBLabelWrapper:
    """Wraps XGBClassifier to handle non-zero-based integer labels.

    XGBoost requires labels in [0, n_classes); PetFinder tabular attributes
    are 1-indexed (e.g. FurLength ∈ {1,2,3}). This wrapper encodes on fit
    and inverse-transforms on predict so the rest of the pipeline is unaffected.
    """

    def __init__(self, **kwargs):
        from xgboost import XGBClassifier
        self._clf = XGBClassifier(**kwargs)
        self._le = LabelEncoder()

    def fit(self, X, y):
        self._clf.fit(X, self._le.fit_transform(y))
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._clf.predict(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    @property
    def classes_(self):
        return self._le.classes_


def build_classifier(name: str, gpu: bool = False, random_state: int = 42, **kwargs):
    """Return a fresh, configured, sklearn-compatible classifier.

    Args:
        name: one of ``SUPPORTED``.
        gpu: enable GPU acceleration where the backend supports it.
        random_state: fixed seed for reproducibility.
        **kwargs: forwarded to the underlying constructor, overriding defaults.

    Returns:
        An unfitted sklearn-compatible classifier with ``fit`` / ``predict`` /
        ``predict_proba`` methods.
    """
    name = name.lower()

    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        defaults = dict(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            device="gpu" if gpu else "cpu",
        )
        return LGBMClassifier(**{**defaults, **kwargs})

    if name == "xgboost":
        defaults = dict(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            device="cuda" if gpu else "cpu",
            eval_metric="mlogloss",
        )
        return _XGBLabelWrapper(**{**defaults, **kwargs})

    if name == "catboost":
        from catboost import CatBoostClassifier
        defaults = dict(
            iterations=100,
            random_seed=random_state,
            verbose=0,
            task_type="GPU" if gpu else "CPU",
        )
        return CatBoostClassifier(**{**defaults, **kwargs})

    if name == "decision_tree":
        defaults = dict(
            random_state=random_state,
            max_depth=kwargs.pop("max_depth", 10),
        )
        return DecisionTreeClassifier(**{**defaults, **kwargs})

    if name == "svm":
        defaults = dict(
            probability=True,     # required for predict_proba in Stage 1
            random_state=random_state,
            kernel=kwargs.pop("kernel", "rbf"),
            C=kwargs.pop("C", 1.0),
        )
        return SVC(**{**defaults, **kwargs})

    raise ValueError(f"Unknown classifier '{name}'. Choose from: {SUPPORTED}")


def build_stage2_classifier(name: str, gpu: bool = False, random_state: int = 42):
    """Return a Stage-2 classifier with stronger defaults suited to AdoptionSpeed.

    Stage 2 has more features (860-dim) and a smaller label space (5 classes),
    so tree-based models can use more estimators / deeper trees.
    """
    overrides: dict = {}
    if name in ("lightgbm",):
        overrides = dict(n_estimators=200, learning_rate=0.05, max_depth=5)
    elif name == "xgboost":
        overrides = dict(n_estimators=200, learning_rate=0.05, max_depth=5)
    elif name == "catboost":
        overrides = dict(iterations=200, learning_rate=0.05, depth=5)
    elif name == "decision_tree":
        overrides = dict(max_depth=15)
    return build_classifier(name, gpu=gpu, random_state=random_state, **overrides)
