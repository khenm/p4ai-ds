import json
import logging
import os
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.text_classification.traditional_ml import (
    ARTIFACT_DIR,
    TEXT_BERT_DIR,
    TEXT_TRADITIONAL_ML_DIR,
    _confusion_for_top_labels,
    _sample_predictions,
    _save_json,
    _top_confusions,
    build_text_splits,
)
from src.utils.env import seed_everything


logger = logging.getLogger(__name__)


def _save_bert_json(data, filename):
    return _save_json(data, filename, output_dir=TEXT_BERT_DIR)


class TextPairDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.labels[idx]
        return encoded


class TransformerPoolingClassifier(torch.nn.Module):
    """Transformer encoder + explicit pooling + linear classification head."""

    def __init__(self, model_name, num_labels, pooling="cls", dropout=0.2):
        super().__init__()
        hf = _load_transformers()
        AutoModel = hf["AutoModel"]
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def _pool(self, outputs, attention_mask):
        hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts

        if self.pooling == "pooler":
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is not None:
                return pooler_output
            logger.warning("Model has no pooler_output; falling back to CLS pooling.")

        return hidden[:, 0]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **_):
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        pooled = self._pool(outputs, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return SimpleNamespace(logits=logits)


def _load_transformers():
    try:
        from transformers import AutoModel
        from transformers import AutoTokenizer
        from transformers import DataCollatorWithPadding
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError(
            "Transformer text training requires `transformers`. Install dependencies with `uv sync` "
            "after updating `pyproject.toml`."
        ) from exc

    return {
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def _metrics_dict(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def _evaluate_model(model, loader, device, loss_fn, stage_name="eval"):
    model.eval()
    predictions = []
    references = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        progress = tqdm(loader, desc=stage_name, leave=False)
        for batch in progress:
            labels = batch["labels"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            total_loss += float(loss.item())
            steps += 1
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().tolist())
            references.extend(labels.cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = _metrics_dict(references, predictions)
    metrics["loss"] = round(total_loss / max(steps, 1), 4)
    return metrics, predictions, references


def _train_epoch(model, loader, optimizer, scheduler, device, loss_fn, epoch, epochs):
    model.train()
    total_loss = 0.0
    steps = 0

    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=False)
    for batch in progress:
        labels = batch["labels"].to(device)
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return round(total_loss / max(steps, 1), 4)


def _class_weights(labels, num_labels):
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    weights = []
    for idx in range(num_labels):
        count = counts.get(idx, 1)
        weights.append(total / (num_labels * count))
    return torch.tensor(weights, dtype=torch.float32)


def _optional_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _safe_id(value):
    return str(value).replace("/", "__").replace(" ", "_").replace("-", "_")


def _split_csv(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def run_text_bert_classification(
    random_state=42,
    test_size=0.2,
    val_size=0.1,
    sample_size=None,
    model_name="distilbert-base-uncased",
    pooling="cls",
    dropout=0.2,
    freeze_encoder=False,
    max_length=128,
    epochs=2,
    batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    output_prefix=None,
):
    seed_everything(random_state)
    logger.info("Preparing transformer text pipeline with model `%s`", model_name)
    hf = _load_transformers()
    AutoTokenizer = hf["AutoTokenizer"]
    DataCollatorWithPadding = hf["DataCollatorWithPadding"]
    get_linear_schedule_with_warmup = hf["get_linear_schedule_with_warmup"]

    logger.info("Building train/val/test splits...")
    splits = build_text_splits(
        test_size=test_size,
        val_size=val_size,
        sample_size=sample_size,
        random_state=random_state,
    )

    x_train = splits["x_train"]
    y_train = splits["y_train"]
    x_val = splits["x_val"]
    y_val = splits["y_val"]
    x_test = splits["x_test"]
    y_test = splits["y_test"]

    labels = sorted(y_train.unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    y_train_ids = y_train.map(label2id)
    y_val_ids = y_val.map(label2id)
    y_test_ids = y_test.map(label2id)

    logger.info(
        "Split sizes | train=%s val=%s test=%s classes=%s",
        len(x_train),
        len(x_val),
        len(x_test),
        len(labels),
    )
    logger.info("Loading tokenizer from `%s`...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(
        "Loading transformer encoder from `%s` with pooling=`%s` and a linear classifier...",
        model_name,
        pooling,
    )
    model = TransformerPoolingClassifier(
        model_name=model_name,
        num_labels=len(labels),
        pooling=pooling,
        dropout=dropout,
    )
    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        logger.info("Encoder is frozen; training only pooling classifier head.")
    logger.info(
        "Pipeline: text -> tokenize -> transformer encoder -> %s pooling -> linear classifier.",
        pooling,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dataset = TextPairDataset(x_train, y_train_ids, tokenizer, max_length=max_length)
    val_dataset = TextPairDataset(x_val, y_val_ids, tokenizer, max_length=max_length)
    test_dataset = TextPairDataset(x_test, y_test_ids, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Training on device: %s", device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    logger.info(
        "Optimization setup | epochs=%s batch_size=%s total_steps=%s warmup_steps=%s",
        epochs,
        batch_size,
        total_steps,
        warmup_steps,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=_class_weights(y_train_ids, len(labels)).to(device)
    )

    best_state = None
    best_val_macro_f1 = -1.0
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, epochs)
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn, epoch, epochs)
        logger.info("Running validation for epoch %s/%s", epoch, epochs)
        val_metrics, _, _ = _evaluate_model(model, val_loader, device, loss_fn, stage_name=f"val {epoch}/{epochs}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
        )
        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_macro_f1=%.4f",
            epoch,
            epochs,
            train_loss,
            val_metrics["macro_f1"],
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_seconds = round(float(time.perf_counter() - started), 2)
    logger.info("Evaluating best checkpoint on the test split...")
    test_metrics, test_predictions, test_references = _evaluate_model(model, test_loader, device, loss_fn, stage_name="test")
    test_metrics["train_seconds"] = train_seconds

    report = classification_report(
        test_references,
        test_predictions,
        target_names=[id2label[idx] for idx in range(len(labels))],
        output_dict=True,
        zero_division=0,
    )
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

    predicted_labels = [id2label[idx] for idx in test_predictions]
    reference_labels = [id2label[idx] for idx in test_references]
    top_labels = y_train.value_counts().head(10).index.tolist()

    safe_model_name = _safe_id(model_name)
    mode_name = "frozen" if freeze_encoder else "finetune"
    pipeline_id = f"{safe_model_name}_{pooling}_{mode_name}"
    output_prefix = output_prefix or f"text_transformer_{pipeline_id}"
    model_dir = os.path.join(ARTIFACT_DIR, pipeline_id)
    os.makedirs(model_dir, exist_ok=True)
    model.encoder.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "pooling": pooling,
            "dropout": dropout,
            "freeze_encoder": freeze_encoder,
            "model_name": model_name,
            "label2id": label2id,
            "id2label": id2label,
        },
        os.path.join(model_dir, "pooling_classifier.pt"),
    )
    with open(os.path.join(model_dir, "train_history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    overview = {
        "dataset": "News_Category_Dataset_v3.json",
        "task": "multi_class_text_classification",
        "model_family": "transformer",
        "pipeline_id": pipeline_id,
        "model_name": model_name,
        "architecture": "tokenizer_encoder_pooling_linear",
        "pooling": pooling,
        "dropout": dropout,
        "freeze_encoder": freeze_encoder,
        "train_size": int(len(x_train)),
        "val_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "class_count": int(len(labels)),
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "class_weighting": True,
        "device": str(device),
        **test_metrics,
    }

    _save_bert_json(overview, f"{output_prefix}_overview.json")
    _save_bert_json({"rows": history}, f"{output_prefix}_training_curve.json")
    _save_bert_json(
        {
            "pipeline_id": pipeline_id,
            "model_name": model_name,
            "pooling": pooling,
            "freeze_encoder": freeze_encoder,
            "accuracy": round(float(report["accuracy"]), 4),
            "macro_avg": {
                "precision": round(float(report["macro avg"]["precision"]), 4),
                "recall": round(float(report["macro avg"]["recall"]), 4),
                "f1_score": round(float(report["macro avg"]["f1-score"]), 4),
            },
            "weighted_avg": {
                "precision": round(float(report["weighted avg"]["precision"]), 4),
                "recall": round(float(report["weighted avg"]["recall"]), 4),
                "f1_score": round(float(report["weighted avg"]["f1-score"]), 4),
            },
            "per_class": per_class,
        },
        f"{output_prefix}_classification_report.json",
    )
    _save_bert_json(
        {
            "pipeline_id": pipeline_id,
            "model_name": model_name,
            "pooling": pooling,
            "freeze_encoder": freeze_encoder,
            "labels": top_labels,
            "matrix": _confusion_for_top_labels(reference_labels, predicted_labels, top_labels, normalize=True),
            "top_confusions": _top_confusions(reference_labels, predicted_labels),
        },
        f"{output_prefix}_confusion_matrix.json",
    )
    _save_bert_json(
        {
            "pipeline_id": pipeline_id,
            "model_name": model_name,
            "pooling": pooling,
            "freeze_encoder": freeze_encoder,
            "rows": _sample_predictions(x_test, y_test, np.array(predicted_labels), limit=12),
        },
        f"{output_prefix}_error_samples.json",
    )
    _save_bert_json(
        {
            "pipeline_id": pipeline_id,
            "model_name": model_name,
            "pooling": pooling,
            "freeze_encoder": freeze_encoder,
            "artifact_dir": model_dir,
            "label2id": label2id,
        },
        f"{output_prefix}_artifact.json",
    )

    baseline_overview = _optional_json(
        os.path.join(TEXT_TRADITIONAL_ML_DIR, "text_ml_overview.json")
    )

    comparison_rows = []
    if baseline_overview is not None:
        comparison_rows.append(
            {
                "model": baseline_overview["best_model"]["name"],
                "family": "traditional_ml",
                "accuracy": baseline_overview["best_model"]["accuracy"],
                "macro_f1": baseline_overview["best_model"]["macro_f1"],
                "weighted_f1": baseline_overview["best_model"]["weighted_f1"],
                "macro_precision": baseline_overview["best_model"]["macro_precision"],
                "macro_recall": baseline_overview["best_model"]["macro_recall"],
                "train_seconds": baseline_overview["best_model"]["train_seconds"],
            }
        )

    comparison_rows.append(
        {
            "model": pipeline_id,
            "family": "transformer",
            "model_name": model_name,
            "pooling": pooling,
            "freeze_encoder": freeze_encoder,
            **test_metrics,
        }
    )
    _save_bert_json({"rows": comparison_rows}, f"{output_prefix}_comparison.json")

    logger.info("%s test macro_f1=%.4f | accuracy=%.4f", pipeline_id, test_metrics["macro_f1"], test_metrics["accuracy"])
    return {
        "pipeline_id": pipeline_id,
        "model_name": model_name,
        "pooling": pooling,
        "freeze_encoder": freeze_encoder,
        "metrics": test_metrics,
        "artifact_dir": model_dir,
    }


def run_transformer_text_pipeline_grid(
    random_state=42,
    test_size=0.2,
    val_size=0.1,
    sample_size=None,
    model_names=None,
    poolings=None,
    freeze_modes=None,
    max_length=128,
    epochs=2,
    batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    dropout=0.2,
    limit=None,
):
    model_names = model_names or [
        "distilbert-base-uncased",
        "bert-base-uncased",
    ]
    poolings = poolings or ["cls", "mean", "pooler"]
    freeze_modes = freeze_modes or [False]

    combinations = [
        (model_name, pooling, freeze_encoder)
        for model_name in model_names
        for pooling in poolings
        for freeze_encoder in freeze_modes
    ]
    if limit is not None:
        combinations = combinations[:limit]

    logger.info("Transformer grid pipelines: %s", len(combinations))
    logger.info("Step 1 model/tokenizer choices: %s", model_names)
    logger.info("Step 2 pooling choices: %s", poolings)
    logger.info("Step 3 classifier: dropout + linear head")
    logger.info("Train modes: %s", ["frozen" if mode else "finetune" for mode in freeze_modes])

    rows = []
    for model_name, pooling, freeze_encoder in tqdm(combinations, desc="Transformer pipeline grid", unit="pipeline"):
        safe_model_name = _safe_id(model_name)
        mode_name = "frozen" if freeze_encoder else "finetune"
        output_prefix = f"text_transformer_{safe_model_name}_{pooling}_{mode_name}"
        try:
            result = run_text_bert_classification(
                random_state=random_state,
                test_size=test_size,
                val_size=val_size,
                sample_size=sample_size,
                model_name=model_name,
                pooling=pooling,
                dropout=dropout,
                freeze_encoder=freeze_encoder,
                max_length=max_length,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                output_prefix=output_prefix,
            )
            row = {
                "pipeline_id": result["pipeline_id"],
                "model_name": model_name,
                "pooling": pooling,
                "freeze_encoder": freeze_encoder,
                "status": "ok",
                **result["metrics"],
                "artifact_dir": result["artifact_dir"],
            }
        except Exception as exc:
            logger.exception("Transformer pipeline failed: %s | %s | freeze=%s", model_name, pooling, freeze_encoder)
            row = {
                "pipeline_id": f"{safe_model_name}_{pooling}_{mode_name}",
                "model_name": model_name,
                "pooling": pooling,
                "freeze_encoder": freeze_encoder,
                "status": "failed",
                "error": repr(exc),
            }

        rows.append(row)
        _save_bert_json({"rows": rows}, "text_transformer_grid_comparison_partial.json")

    rows.sort(
        key=lambda row: (
            row.get("macro_f1", -1) if row.get("macro_f1") is not None else -1,
            row.get("weighted_f1", -1) if row.get("weighted_f1") is not None else -1,
        ),
        reverse=True,
    )
    overview = {
        "dataset": "News_Category_Dataset_v3.json",
        "task": "multi_class_text_classification",
        "workflow": "tokenizer_encoder_pooling_linear_grid",
        "sample_size": sample_size,
        "model_names": model_names,
        "poolings": poolings,
        "freeze_modes": freeze_modes,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "best_pipeline": rows[0] if rows else None,
    }
    _save_bert_json({"rows": rows}, "text_transformer_grid_comparison.json")
    _save_bert_json(overview, "text_transformer_grid_overview.json")

    baseline_overview = _optional_json(
        os.path.join(TEXT_TRADITIONAL_ML_DIR, "text_ml_overview.json")
    )
    comparison_rows = []
    if baseline_overview is not None:
        comparison_rows.append(
            {
                "model": baseline_overview["best_model"]["name"],
                "family": "traditional_ml",
                "accuracy": baseline_overview["best_model"]["accuracy"],
                "macro_f1": baseline_overview["best_model"]["macro_f1"],
                "weighted_f1": baseline_overview["best_model"]["weighted_f1"],
                "macro_precision": baseline_overview["best_model"]["macro_precision"],
                "macro_recall": baseline_overview["best_model"]["macro_recall"],
                "train_seconds": baseline_overview["best_model"]["train_seconds"],
            }
        )
    for row in rows:
        if row["status"] == "ok":
            comparison_rows.append(
                {
                    "model": row["pipeline_id"],
                    "family": "transformer",
                    "model_name": row["model_name"],
                    "pooling": row["pooling"],
                    "freeze_encoder": row["freeze_encoder"],
                    "accuracy": row["accuracy"],
                    "macro_f1": row["macro_f1"],
                    "weighted_f1": row["weighted_f1"],
                    "macro_precision": row["macro_precision"],
                    "macro_recall": row["macro_recall"],
                    "train_seconds": row["train_seconds"],
                }
            )
    _save_bert_json({"rows": comparison_rows}, "text_assignment2_model_comparison.json")

    return {
        "best_pipeline": rows[0] if rows else None,
        "rows": rows,
    }
