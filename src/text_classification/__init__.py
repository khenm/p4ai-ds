"""Text classification pipelines for Assignment 2."""

from src.text_classification.pipeline_grid import run_text_pipeline_grid
from src.text_classification.traditional_ml import (
    ARTIFACT_DIR,
    TEXT_BERT_DIR,
    TEXT_PIPELINE_GRID_DIR,
    TEXT_TRADITIONAL_ML_DIR,
    LabelEncodedClassifier,
    build_text_splits,
    run_text_classification,
)

__all__ = [
    "ARTIFACT_DIR",
    "TEXT_BERT_DIR",
    "TEXT_PIPELINE_GRID_DIR",
    "TEXT_TRADITIONAL_ML_DIR",
    "LabelEncodedClassifier",
    "build_text_splits",
    "run_text_classification",
    "run_text_pipeline_grid",
]
