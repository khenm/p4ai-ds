import json
import logging
import os
import pickle
import time

from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from tqdm.auto import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.eda.text_context import NpEncoder, load_text_dataset
try:
    from src.utils.env import seed_everything
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    def seed_everything(seed=42):
        np.random.seed(seed)


logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "text_classification")
TEXT_UI_DATA_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "data", "text_classification")
ARTIFACT_DIR = os.path.join(TEXT_RESULTS_DIR, "artifacts")
TEXT_TRADITIONAL_ML_DIR = os.path.join(TEXT_RESULTS_DIR, "traditional_ml")
TEXT_PIPELINE_GRID_DIR = os.path.join(TEXT_RESULTS_DIR, "pipeline_grid")
TEXT_BERT_DIR = os.path.join(TEXT_RESULTS_DIR, "bert")
for output_dir in (ARTIFACT_DIR, TEXT_TRADITIONAL_ML_DIR, TEXT_PIPELINE_GRID_DIR, TEXT_BERT_DIR):
    os.makedirs(output_dir, exist_ok=True)
USE_XGBOOST_GPU = os.getenv("TEXT_XGBOOST_GPU", "").strip().lower() in {"1", "true", "yes", "cuda"}


def _mirror_output_dir(output_dir):
    output_dir = os.path.abspath(output_dir)
    results_root = os.path.abspath(TEXT_RESULTS_DIR)
    if os.path.commonpath([output_dir, results_root]) != results_root:
        return None
    rel_path = os.path.relpath(output_dir, results_root)
    return os.path.join(TEXT_UI_DATA_DIR, rel_path)


def _save_json(data, filename, output_dir=TEXT_TRADITIONAL_ML_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, cls=NpEncoder)

    mirror_dir = _mirror_output_dir(output_dir)
    if mirror_dir is not None:
        os.makedirs(mirror_dir, exist_ok=True)
        with open(os.path.join(mirror_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, cls=NpEncoder)
    return path


def _prepare_text_frame(min_samples_per_class=2, sample_size=None, random_state=42):
    df = load_text_dataset()
    df = df.loc[df["combined_text"].str.strip().ne("")].copy()

    class_counts = df["category"].value_counts()
    keep_labels = class_counts[class_counts >= min_samples_per_class].index
    df = df[df["category"].isin(keep_labels)].copy()

    if sample_size is not None and sample_size < len(df):
        sampled = []
        total_rows = len(df)
        for _, group in df.groupby("category", group_keys=False):
            n_take = max(2, round(len(group) / total_rows * sample_size))
            sampled.append(group.sample(n=min(len(group), n_take), random_state=random_state))
        df = (
            np.random.default_rng(random_state)
            .permutation(np.concatenate([group.index.to_numpy() for group in sampled]))
        )
        df = load_text_dataset().loc[df].copy()
        counts = df["category"].value_counts()
        df = df[df["category"].isin(counts[counts >= 2].index)].copy()

    df["text_length_words"] = df["combined_text"].str.split().str.len()
    return df


class SafeTruncatedSVD(BaseEstimator, TransformerMixin):
    """Cap SVD dimensions for small samples where TF-IDF has few features."""

    def __init__(self, n_components=300, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, x, y=None):
        n_features = x.shape[1]
        if n_features <= 1:
            self.svd_ = None
            self.n_components_ = n_features
            return self

        self.n_components_ = min(self.n_components, n_features - 1)
        self.svd_ = TruncatedSVD(n_components=self.n_components_, random_state=self.random_state)
        self.svd_.fit(x, y)
        return self

    def transform(self, x):
        if self.svd_ is None:
            return x.toarray() if hasattr(x, "toarray") else x
        return self.svd_.transform(x)


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Adapter for classifiers such as XGBoost that expect numeric class labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, y_encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, x):
        encoded = self.estimator_.predict(x).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, x):
        return self.estimator_.predict_proba(x)


def prepare_text_frame(min_samples_per_class=2, sample_size=None, random_state=42):
    return _prepare_text_frame(
        min_samples_per_class=min_samples_per_class,
        sample_size=sample_size,
        random_state=random_state,
    )


def build_text_splits(test_size=0.2, val_size=0.1, sample_size=None, random_state=42):
    df = prepare_text_frame(sample_size=sample_size, random_state=random_state)
    x = df["combined_text"]
    y = df["category"]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if val_size and val_size > 0:
        adjusted_val_size = val_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=y_train_full,
        )
    else:
        x_train, y_train = x_train_full, y_train_full
        x_val = y_val = None

    return {
        "df": df,
        "x_train": x_train.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "x_val": None if x_val is None else x_val.reset_index(drop=True),
        "y_val": None if y_val is None else y_val.reset_index(drop=True),
        "x_test": x_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


def _build_models(random_state, estimator_n_jobs=-1):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    def sparse_pipeline(clf):
        return Pipeline(
            [
                ("tfidf", clone(vectorizer)),
                ("clf", clf),
            ]
        )

    def char_pipeline(clf):
        return Pipeline(
            [
                ("tfidf", clone(char_vectorizer)),
                ("clf", clf),
            ]
        )

    def lsa_pipeline(clf):
        return Pipeline(
            [
                ("tfidf", clone(vectorizer)),
                ("svd", SafeTruncatedSVD(n_components=300, random_state=random_state)),
                ("clf", clf),
            ]
        )

    models = {
        "multinomial_nb": sparse_pipeline(MultinomialNB(alpha=0.15)),
        "complement_nb": sparse_pipeline(ComplementNB(alpha=0.15)),
        "logistic_regression": Pipeline(
            [
                ("tfidf", clone(vectorizer)),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            max_iter=300,
                            solver="liblinear",
                            class_weight="balanced",
                            tol=1e-4,
                            random_state=random_state,
                        ),
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "linear_svc": sparse_pipeline(
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                random_state=random_state,
            )
        ),
        "passive_aggressive": sparse_pipeline(
            SGDClassifier(
                loss="hinge",
                penalty=None,
                learning_rate="pa1",
                eta0=1.0,
                max_iter=50,
                tol=1e-3,
                class_weight="balanced",
                n_jobs=estimator_n_jobs,
                random_state=random_state,
            )
        ),
        "ridge_classifier": sparse_pipeline(
            RidgeClassifier(
                alpha=1.0,
                class_weight="balanced",
            )
        ),
        "char_linear_svc": char_pipeline(
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                random_state=random_state,
            )
        ),
        "sgd_log_loss": sparse_pipeline(
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                max_iter=30,
                tol=1e-3,
                class_weight="balanced",
                n_jobs=estimator_n_jobs,
                random_state=random_state,
            )
        ),
        "sgd_hinge": sparse_pipeline(
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=1e-5,
                max_iter=30,
                tol=1e-3,
                class_weight="balanced",
                n_jobs=estimator_n_jobs,
                random_state=random_state,
            )
        ),
        "hashing_sgd_hinge": Pipeline(
            [
                (
                    "hashing",
                    HashingVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        n_features=2**20,
                        alternate_sign=False,
                        norm=None,
                        dtype=np.float32,
                    ),
                ),
                ("tfidf_transform", TfidfTransformer(sublinear_tf=True)),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=1e-5,
                        max_iter=30,
                        tol=1e-3,
                        class_weight="balanced",
                        n_jobs=estimator_n_jobs,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "voting_ensemble": Pipeline(
            [
                ("tfidf", clone(vectorizer)),
                (
                    "clf",
                    VotingClassifier(
                        estimators=[
                            (
                                "lr",
                                OneVsRestClassifier(
                                    LogisticRegression(
                                        max_iter=300,
                                        solver="liblinear",
                                        class_weight="balanced",
                                        tol=1e-4,
                                        random_state=random_state,
                                    ),
                                    n_jobs=1,
                                ),
                            ),
                            (
                                "svm",
                                LinearSVC(
                                    class_weight="balanced",
                                    dual="auto",
                                    random_state=random_state,
                                ),
                            ),
                            ("nb", ComplementNB(alpha=0.15)),
                        ],
                        voting="hard",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        xgb_kwargs = {
            "n_estimators": 160,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": estimator_n_jobs,
            "random_state": random_state,
        }
        if USE_XGBOOST_GPU:
            xgb_kwargs["device"] = "cuda"

        models["xgboost"] = lsa_pipeline(
            LabelEncodedClassifier(
                XGBClassifier(**xgb_kwargs)
            )
        )
    else:
        logger.warning("XGBoost is not installed; skipping the xgboost text classifier.")

    return models


def _evaluate_named_model(model_name, model, x_train, y_train, x_test, y_test):
    return model_name, _evaluate_model(
        model_name,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
    )


def _evaluate_model(model_name, model, x_train, y_train, x_test, y_test):
    started = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = round(float(time.perf_counter() - started), 2)
    pred = model.predict(x_test)

    return {
        "pipeline": model,
        "predictions": pred,
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_test, pred, average="weighted")), 4),
            "macro_precision": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
            "train_seconds": train_seconds,
        },
    }


def _evaluate_models(models, x_train, y_train, x_test, y_test, n_jobs=1):
    model_items = list(models.items())
    progress_kwargs = {
        "total": len(model_items),
        "desc": "Training text models",
        "unit": "model",
        "dynamic_ncols": True,
    }

    if n_jobs == 1:
        results = {}
        with tqdm(**progress_kwargs) as progress:
            for model_name, pipeline in model_items:
                progress.set_postfix(model=model_name)
                name, result = _evaluate_named_model(model_name, pipeline, x_train, y_train, x_test, y_test)
                results[name] = result
                progress.update(1)
        return results

    tasks = (
        delayed(_evaluate_named_model)(model_name, pipeline, x_train, y_train, x_test, y_test)
        for model_name, pipeline in model_items
    )
    results = {}
    parallel = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator_unordered")
    with tqdm(**progress_kwargs) as progress:
        for model_name, result in parallel(tasks):
            progress.set_postfix(model=model_name)
            results[model_name] = result
            progress.update(1)
    return results


def _top_confusions(y_true, y_pred, limit=12):
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    payload = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i == j or matrix[i, j] == 0:
                continue
            payload.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[i, j]),
                }
            )
    payload.sort(key=lambda row: row["count"], reverse=True)
    return payload[:limit]


def _confusion_for_top_labels(y_true, y_pred, top_labels, normalize=True):
    matrix = confusion_matrix(y_true, y_pred, labels=top_labels)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )
    return [[round(float(cell), 4) for cell in row] for row in matrix.tolist()]


def _extract_feature_scores(pipeline, labels, selected_labels, top_k=12):
    clf = pipeline.named_steps["clf"]
    tfidf = pipeline.named_steps.get("tfidf")
    if tfidf is None:
        return []
    features = tfidf.get_feature_names_out()

    if hasattr(clf, "coef_"):
        scores = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        scores = clf.feature_log_prob_
    elif hasattr(clf, "estimators_"):
        scores = None
    else:
        return []

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    payload = []
    for label in selected_labels:
        idx = label_to_idx.get(label)
        if idx is None:
            continue

        if scores is not None and idx < scores.shape[0]:
            row = scores[idx]
        elif hasattr(clf, "estimators_") and idx < len(clf.estimators_) and hasattr(clf.estimators_[idx], "coef_"):
            row = clf.estimators_[idx].coef_.ravel()
        else:
            continue

        top_idx = row.argsort()[::-1][:top_k]
        payload.append(
            {
                "category": label,
                "features": [
                    {"term": str(features[i]), "score": round(float(row[i]), 4)}
                    for i in top_idx
                ],
            }
        )
    return payload


def _sample_predictions(x_test, y_test, y_pred, limit=12):
    samples = []
    for text, actual, predicted in zip(x_test.tolist(), y_test.tolist(), y_pred.tolist()):
        if actual == predicted:
            continue
        samples.append(
            {
                "actual": actual,
                "predicted": predicted,
                "text_preview": text[:240],
            }
        )
        if len(samples) >= limit:
            break
    return samples


def run_text_classification(random_state=42, test_size=0.2, sample_size=None, n_jobs=1):
    seed_everything(random_state)
    splits = build_text_splits(
        test_size=test_size,
        val_size=0.0,
        sample_size=sample_size,
        random_state=random_state,
    )
    df = splits["df"]
    x_train = splits["x_train"]
    y_train = splits["y_train"]
    x_test = splits["x_test"]
    y_test = splits["y_test"]

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    top_labels = train_counts.head(10).index.tolist()

    estimator_n_jobs = -1 if n_jobs == 1 else 1
    models = _build_models(random_state, estimator_n_jobs=estimator_n_jobs)
    model_results = _evaluate_models(
        models,
        x_train,
        y_train,
        x_test,
        y_test,
        n_jobs=n_jobs,
    )

    best_model_name, best_result = max(
        model_results.items(),
        key=lambda item: (item[1]["metrics"]["macro_f1"], item[1]["metrics"]["weighted_f1"]),
    )
    best_pipeline = best_result["pipeline"]
    best_pred = best_result["predictions"]

    report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    per_class = []
    for label, stats in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        per_class.append(
            {
                "category": label,
                "precision": round(float(stats["precision"]), 4),
                "recall": round(float(stats["recall"]), 4),
                "f1_score": round(float(stats["f1-score"]), 4),
                "support": int(stats["support"]),
            }
        )
    per_class.sort(key=lambda row: row["support"], reverse=True)

    summary = {
        "dataset": "News_Category_Dataset_v3.json",
        "task": "multi_class_text_classification",
        "workflow": "scalable_ml_only",
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "class_count": int(df["category"].nunique()),
        "avg_words_per_article": round(float(df["text_length_words"].mean()), 2),
        "median_words_per_article": int(df["text_length_words"].median()),
        "class_imbalance_ratio": round(float(train_counts.max() / train_counts.min()), 2),
        "tfidf_config": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "min_df": 5,
            "max_df": 0.95,
            "max_features": 50000,
            "hashing_features": 2**20,
            "sublinear_tf": True,
            "stop_words": "english",
        },
        "eda_guidance": [
            "Combined headline and short description are used to increase text context while staying lightweight.",
            "Macro-F1 is treated as the primary ranking metric because the dataset is imbalanced across categories.",
            "The model set avoids RBF kernels, dense sklearn boosting, and MLP baselines because they do not scale well on the full text dataset.",
            "Linear, Naive Bayes, SGD, character n-gram, hashing, voting, and histogram XGBoost baselines are used for faster CPU/GPU-friendly experiments.",
        ],
        "best_model": {
            "name": best_model_name,
            **best_result["metrics"],
        },
    }

    comparison_rows = []
    for model_name, result in model_results.items():
        comparison_rows.append({"model": model_name, **result["metrics"]})
    comparison_rows.sort(key=lambda row: (row["macro_f1"], row["weighted_f1"]), reverse=True)

    metrics_summary = report["macro avg"]
    weighted_summary = report["weighted avg"]
    labels = list(best_pipeline.named_steps["clf"].classes_)

    _save_json(summary, "text_ml_overview.json")
    _save_json({"rows": comparison_rows}, "text_ml_model_comparison.json")
    _save_json(
        {
            "model": best_model_name,
            "accuracy": round(float(report["accuracy"]), 4),
            "macro_avg": {
                "precision": round(float(metrics_summary["precision"]), 4),
                "recall": round(float(metrics_summary["recall"]), 4),
                "f1_score": round(float(metrics_summary["f1-score"]), 4),
            },
            "weighted_avg": {
                "precision": round(float(weighted_summary["precision"]), 4),
                "recall": round(float(weighted_summary["recall"]), 4),
                "f1_score": round(float(weighted_summary["f1-score"]), 4),
            },
            "per_class": per_class,
        },
        "text_ml_classification_report.json",
    )
    _save_json(
        {
            "model": best_model_name,
            "labels": top_labels,
            "matrix": _confusion_for_top_labels(y_test, best_pred, top_labels, normalize=True),
            "top_confusions": _top_confusions(y_test, best_pred),
        },
        "text_ml_confusion_matrix.json",
    )
    _save_json(
        {
            "model": best_model_name,
            "top_labels": top_labels[:6],
            "groups": _extract_feature_scores(best_pipeline, labels, top_labels[:6]),
        },
        "text_ml_top_features.json",
    )
    _save_json(
        {
            "model": best_model_name,
            "rows": _sample_predictions(x_test.reset_index(drop=True), y_test.reset_index(drop=True), best_pred),
        },
        "text_ml_error_samples.json",
    )
    _save_json(
        {
            "train_class_distribution": [
                {"category": label, "count": int(count)}
                for label, count in train_counts.head(15).items()
            ],
            "test_class_distribution": [
                {"category": label, "count": int(count)}
                for label, count in test_counts.head(15).items()
            ],
        },
        "text_ml_split_summary.json",
    )

    model_path = os.path.join(ARTIFACT_DIR, "best_text_model.pkl")
    with open(model_path, "wb") as handle:
        pickle.dump(best_pipeline, handle)

    _save_json(
        {
            "best_model_name": best_model_name,
            "artifact_path": model_path,
        },
        "text_ml_artifact.json",
    )

    logger.info("Best text model: %s", best_model_name)
    logger.info("Macro-F1: %.4f | Weighted-F1: %.4f", best_result["metrics"]["macro_f1"], best_result["metrics"]["weighted_f1"])

    return {
        "best_model_name": best_model_name,
        "metrics": best_result["metrics"],
        "artifact_path": model_path,
    }
