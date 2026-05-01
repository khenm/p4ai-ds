"""Factory for interchangeable sklearn-compatible ML classifiers.

Supported names
---------------
lightgbm      LGBMClassifier      (fast gradient boosting, GPU via OpenCL)
xgboost       XGBClassifier       (gradient boosting, GPU via CUDA/hist)
catboost      CatBoostClassifier  (gradient boosting, GPU native)
decision_tree DecisionTreeClassifier
svm           SVC with probability=True  (slow on large feature sets)
"""

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SUPPORTED = ("lightgbm", "xgboost", "catboost", "decision_tree", "svm")


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
        from xgboost import XGBClassifier
        defaults = dict(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            device="cuda" if gpu else "cpu",
            eval_metric="mlogloss",
        )
        return XGBClassifier(**{**defaults, **kwargs})

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
