import os
import logging
import sys
import argparse
import warnings
import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.petfinder import PetFinderDataset, get_transforms
from src.models.cnn import TwoStageResNet
from src.models.ml_classifiers import SUPPORTED, build_classifier, build_stage2_classifier
from src.utils.env import load_config, seed_everything
from src.utils.reporting import (
    build_adoption_speed_section,
    build_breed_section,
    save_report,
)
from src.analysis.ablation import compute_feature_slices, run_ablation_combinations

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STAGE1_FEATURES = [
    "Type", "FurLength", "MaturitySize", "Breed1",
    "Health", "Vaccinated", "Dewormed", "Sterilized", "Gender", "Color1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML pipeline for PetFinder")
    parser.add_argument("--config", type=str, default="configs/train_lgbm.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=SUPPORTED,
        help="ML backend for Stage 1 and Stage 2 classifiers",
    )
    parser.add_argument(
        "--max_combo_size",
        type=int,
        default=3,
        help="Max feature-group subset size for ablation (0 to skip ablation)",
    )
    return parser.parse_args()


def extract_features(loader, model, device):
    model.eval()
    all_features = []
    all_targets = {k: [] for k in STAGE1_FEATURES + ["AdoptionSpeed", "PhotoAmt"]}

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            for k in all_targets:
                all_targets[k].append(targets[k].numpy())

    return np.vstack(all_features), {k: np.concatenate(v) for k, v in all_targets.items()}


def main():
    args = parse_args()
    config = load_config(args.config)
    model_name = args.model

    seed_everything(config.get("seed_value", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.is_available()
    logger.info(f"Using device: {device} | ML model: {model_name}")

    raw_dir = config["data"]["raw_dir"]
    csv_path = os.path.join(raw_dir, "train", "train.csv")
    img_dir = os.path.join(raw_dir, "train_images")
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 4)

    dataset = PetFinderDataset(csv_path, img_dir, transform=get_transforms(train=False))

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(dataset))
    train_size = int(0.8 * len(dataset))
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    cnn = TwoStageResNet(extract_features=True).to(device)

    logger.info("Extracting features for train set...")
    X_train_img, y_train = extract_features(train_loader, cnn, device)

    logger.info("Extracting features for val set...")
    X_val_img, y_val = extract_features(val_loader, cnn, device)

    # ------------------------------------------------------------------
    # Stage 1: predict each tabular attribute from image embeddings
    # ------------------------------------------------------------------
    logger.info(f"\n--- Stage 1: {model_name} per tabular feature ---")
    models_stage1 = {}
    preds_train_s1, preds_val_s1 = [], []

    for feat in STAGE1_FEATURES:
        logger.info(f"  Training for {feat}...")
        clf = build_classifier(model_name, gpu=gpu)
        clf.fit(X_train_img, y_train[feat])
        preds_train_s1.append(clf.predict_proba(X_train_img))
        preds_val_s1.append(clf.predict_proba(X_val_img))
        models_stage1[feat] = clf
        val_acc = accuracy_score(y_val[feat], clf.predict(X_val_img))
        logger.info(f"    Val Accuracy: {val_acc:.4f}")

    # ------------------------------------------------------------------
    # Stage 2: predict AdoptionSpeed from image + stage-1 logits + PhotoAmt
    # ------------------------------------------------------------------
    logger.info(f"\n--- Stage 2: {model_name} → AdoptionSpeed ---")
    X_train = np.hstack([X_train_img] + preds_train_s1 + [y_train["PhotoAmt"].reshape(-1, 1)])
    X_val   = np.hstack([X_val_img]   + preds_val_s1   + [y_val["PhotoAmt"].reshape(-1, 1)])

    clf_final = build_stage2_classifier(model_name, gpu=gpu)
    clf_final.fit(X_train, y_train["AdoptionSpeed"])

    y_pred = clf_final.predict(X_val)
    acc = accuracy_score(y_val["AdoptionSpeed"], y_pred)
    qwk = cohen_kappa_score(y_val["AdoptionSpeed"], y_pred, weights="quadratic")
    logger.info(f"Final Val — Accuracy: {acc:.4f} | QWK: {qwk:.4f}")

    # ------------------------------------------------------------------
    # Stage 1 validation accuracies
    # ------------------------------------------------------------------
    stage1_acc = {
        feat: round(accuracy_score(y_val[feat], models_stage1[feat].predict(X_val_img)), 4)
        for feat in STAGE1_FEATURES
    }

    # ------------------------------------------------------------------
    # Ablation
    # ------------------------------------------------------------------
    ablation_result = None
    if args.max_combo_size > 0:
        logger.info(f"Running ablation (max_combo_size={args.max_combo_size})...")
        feature_slices = compute_feature_slices(512, models_stage1)
        ablation_result = run_ablation_combinations(
            clf_final, X_val, y_val["AdoptionSpeed"], feature_slices,
            max_combo_size=args.max_combo_size,
        )
        logger.info(f"  {ablation_result['n_combinations']} combinations evaluated.")
        logger.info(f"  Worst: {ablation_result['worst'][0]}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "qwk": round(qwk, 4),
        "stage1_val_accuracy": stage1_acc,
        "adoption_speed": build_adoption_speed_section(y_val["AdoptionSpeed"], y_pred),
        "breed1": build_breed_section(y_val["Breed1"], models_stage1["Breed1"].predict(X_val_img)),
    }
    if ablation_result is not None:
        report["ablation_combinations"] = ablation_result

    # Save under results/reports/<model_name>.json
    config_stem = f"train_ml_{model_name}"
    dummy_config_path = f"configs/{config_stem}.yaml"
    out_path = save_report(dummy_config_path, report)
    logger.info(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
