import logging
import os
import time

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm


from src.text_classification.traditional_ml import (
    ARTIFACT_DIR,
    TEXT_PIPELINE_GRID_DIR,
    LabelEncodedClassifier,
    _save_json as _save_text_json,
    build_text_splits,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _feature_steps():
    common = {
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_df": 0.95,
        "max_features": 50000,
        "dtype": np.float32,
    }
    return {
        "bow": CountVectorizer(**common),
        "tfidf": TfidfVectorizer(sublinear_tf=True, **common),
    }


def _reducer_steps(random_state):
    return {
        "chi2_k20000": SelectKBest(score_func=chi2, k=20000),
        "svd_100": TruncatedSVD(n_components=100, random_state=random_state),
        "svd_300": TruncatedSVD(n_components=300, random_state=random_state),
    }


def _model_steps(random_state, cpu_jobs):
    models = {
        "logistic_regression": OneVsRestClassifier(
            LogisticRegression(
                max_iter=300,
                solver="liblinear",
                class_weight="balanced",
                tol=1e-4,
                random_state=random_state,
            ),
            n_jobs=1,
        ),
        "linear_svc": LinearSVC(
            class_weight="balanced",
            dual="auto",
            random_state=random_state,
        ),
        "sgd_hinge": SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-5,
            max_iter=30,
            tol=1e-3,
            class_weight="balanced",
            n_jobs=cpu_jobs,
            random_state=random_state,
        ),
    }
    dense_models = {
        "mlp": LabelEncodedClassifier(
            MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                early_stopping=True,
                max_iter=80,
                random_state=random_state,
            )
        ),
    }

    return models, dense_models


def _metrics(y_true, y_pred, train_seconds):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "train_seconds": round(float(train_seconds), 2),
    }


def _fit_transform_step(step_name, step, x_train, y_train, x_test):
    if step is None:
        return x_train, x_test, 0.0

    started = time.perf_counter()
    x_train_out = step.fit_transform(x_train, y_train)
    x_test_out = step.transform(x_test)
    seconds = time.perf_counter() - started
    return x_train_out, x_test_out, seconds


def run_text_pipeline_grid(test_size=0.2, sample_size=None, random_state=42, cpu_jobs=10, limit=None):
    splits = build_text_splits(
        test_size=test_size,
        val_size=0.0,
        sample_size=sample_size,
        random_state=random_state,
    )
    x_train = splits["x_train"]
    y_train = splits["y_train"]
    x_test = splits["x_test"]
    y_test = splits["y_test"]

    feature_steps = _feature_steps()
    reducer_steps = _reducer_steps(random_state)
    linear_model_steps, dense_model_steps = _model_steps(random_state, cpu_jobs)
    total = 0
    for reducer_name in reducer_steps:
        total += len(feature_steps) * len(linear_model_steps)
        if reducer_name.startswith("svd_"):
            total += len(feature_steps) * len(dense_model_steps)
    if limit is not None:
        total = min(total, limit)

    logger.info(
        "Dataset split | train=%s test=%s classes=%s pipelines=%s",
        len(x_train),
        len(x_test),
        y_train.nunique(),
        total,
    )
    logger.info("This reduced-dimension grid uses CPU-compatible ML models. GPU is not required.")

    rows = []
    pipeline_count = 0
    best = None
    best_model = None

    with tqdm(total=total, desc="Pipeline grid", unit="pipeline", dynamic_ncols=True) as grid_progress:
        for feature_name, feature_step in feature_steps.items():
            x_train_features, x_test_features, feature_seconds = _fit_transform_step(
                feature_name,
                clone(feature_step),
                x_train,
                y_train,
                x_test,
            )

            for reducer_name, reducer_step in reducer_steps.items():
                x_train_reduced, x_test_reduced, reducer_seconds = _fit_transform_step(
                    reducer_name,
                    None if reducer_step is None else clone(reducer_step),
                    x_train_features,
                    y_train,
                    x_test_features,
                )

                active_model_steps = dict(linear_model_steps)
                if reducer_name.startswith("svd_"):
                    active_model_steps.update(dense_model_steps)

                for model_name, model_step in active_model_steps.items():
                    if limit is not None and pipeline_count >= limit:
                        break

                    pipeline_name = f"{feature_name}__{reducer_name}__{model_name}"
                    grid_progress.set_postfix(pipeline=pipeline_name)
                    model = clone(model_step)
                    started = time.perf_counter()
                    row = {
                        "pipeline": pipeline_name,
                        "feature_step": feature_name,
                        "reduction_step": reducer_name,
                        "model": model_name,
                        "feature_seconds": round(float(feature_seconds), 2),
                        "reduction_seconds": round(float(reducer_seconds), 2),
                    }
                    try:
                        model.fit(x_train_reduced, y_train)
                        y_pred = model.predict(x_test_reduced)
                        train_seconds = time.perf_counter() - started
                        row.update(_metrics(y_test, y_pred, train_seconds))
                        row["status"] = "ok"
                    except Exception as exc:
                        train_seconds = time.perf_counter() - started
                        logger.exception("Pipeline failed: %s", pipeline_name)
                        row.update(
                            {
                                "accuracy": None,
                                "macro_f1": None,
                                "weighted_f1": None,
                                "macro_precision": None,
                                "macro_recall": None,
                                "train_seconds": round(float(train_seconds), 2),
                                "status": "failed",
                                "error": repr(exc),
                            }
                        )
                    rows.append(row)
                    _save_text_json({"rows": rows}, "text_pipeline_grid_comparison_partial.json")

                    if row["status"] == "ok" and (
                        best is None or (row["macro_f1"], row["weighted_f1"]) > (best["macro_f1"], best["weighted_f1"])
                    ):
                        best = row
                        best_model = Pipeline(
                            [
                                ("feature", clone(feature_step)),
                                ("reduction", clone(reducer_step)),
                                ("model", clone(model_step)),
                            ]
                        )

                    pipeline_count += 1
                    grid_progress.update(1)

                if limit is not None and pipeline_count >= limit:
                    break
            if limit is not None and pipeline_count >= limit:
                break

    rows.sort(
        key=lambda row: (
            row["macro_f1"] if row["macro_f1"] is not None else -1,
            row["weighted_f1"] if row["weighted_f1"] is not None else -1,
        ),
        reverse=True,
    )
    overview = {
        "dataset": "News_Category_Dataset_v3.json",
        "task": "multi_class_text_classification",
        "workflow": "reduced_feature_x_classifier_grid",
        "sample_size": sample_size,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "class_count": int(y_train.nunique()),
        "feature_steps": list(feature_steps.keys()),
        "reduction_steps": list(reducer_steps.keys()),
        "linear_model_steps": list(linear_model_steps.keys()),
        "dense_svd_model_steps": list(dense_model_steps.keys()),
        "best_pipeline": rows[0] if rows else None,
    }
    comparison_path = _save_text_json({"rows": rows}, "text_pipeline_grid_comparison.json")
    overview_path = _save_text_json(overview, "text_pipeline_grid_overview.json")

    model_path = None
    if best_model is not None:
        import pickle

        model_path = os.path.join(ARTIFACT_DIR, "best_text_pipeline_grid_model.pkl")
        best_model.fit(x_train, y_train)
        with open(model_path, "wb") as handle:
            pickle.dump(best_model, handle)

    return {
        "best_pipeline": rows[0] if rows else None,
        "comparison_path": comparison_path,
        "overview_path": overview_path,
        "artifact_path": model_path,
    }
