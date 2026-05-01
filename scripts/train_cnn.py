import os
import logging
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.optim import AdamW
import torch.distributed as tdist
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.petfinder import PetFinderDataset, get_transforms
from src.models.cnn import TwoStageResNet
from src.utils.dist import setup_dist, cleanup_dist, is_main_process
from src.utils.env import load_config, seed_everything
from src.utils.gradnorm import GradNormController
from src.utils.reporting import build_clf_section, build_breed_section, save_report
from src.analysis.gradcam import compute_gradcam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_IMG_MEAN = np.array([0.485, 0.456, 0.406])
_IMG_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(tensor):
    """Denormalize a (C, H, W) image tensor → (H, W, 3) uint8 BGR array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * _IMG_STD + _IMG_MEAN).clip(0, 1)
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr


def save_gradcam_samples(model, val_loader, device, out_dir, n_samples=16,
                          target_key="AdoptionSpeed"):
    """Run Grad-CAM on a sample of val images and save overlays as JPEGs."""
    os.makedirs(out_dir, exist_ok=True)
    collected = 0
    model.eval()

    for images, targets in val_loader:
        if collected >= n_samples:
            break
        batch = min(images.size(0), n_samples - collected)
        img_batch = images[:batch].to(device)

        heatmaps, pred_classes = compute_gradcam(model, img_batch, target_key=target_key)

        for i in range(batch):
            pet_id   = targets["PetID"][i]
            true_cls = targets[target_key][i].item()
            pred_cls = pred_classes[i]

            img_bgr = _denorm(images[i])
            hm = (heatmaps[i] * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)

            fname = f"{pet_id}_true{true_cls}_pred{pred_cls}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), overlay)
            collected += 1

    logger.info(f"Saved {collected} Grad-CAM overlays to {out_dir}")


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip DistributedDataParallel to access the raw module."""
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def _gather_list(lst: list, is_dist: bool) -> list:
    """All-gather a Python list from every rank and flatten into one list."""
    if not is_dist:
        return lst
    gathered = [None] * tdist.get_world_size()
    tdist.all_gather_object(gathered, lst)
    return [item for sublist in gathered for item in sublist]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Two-Stage CNN for PetFinder")
    parser.add_argument('--config', type=str, default='configs/train_cnn.yaml')
    parser.add_argument('--use_gradnorm', action=argparse.BooleanOptionalAction, default=None,
                        help='Override use_gradnorm from config (--use_gradnorm / --no-use_gradnorm)')
    parser.add_argument('--freeze_stage1', action=argparse.BooleanOptionalAction, default=None,
                        help='Override freeze_stage1 from config (--freeze_stage1 / --no-freeze_stage1)')
    parser.add_argument('--distributed', choices=['false', 'ddp', 'fsdp'], default=None,
                        help='Override distributed mode (false / ddp / fsdp)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a saved stage-2 checkpoint; skips training and runs Grad-CAM only')
    return parser.parse_args()


def _save_checkpoint(model: nn.Module, config: dict, stage: int, args) -> str:
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    ckpt_dir = os.path.join("results", "checkpoints", config_stem)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"stage{stage}.pt")
    torch.save(_unwrap(model).state_dict(), path)
    logger.info(f"Checkpoint saved to {path}")
    return path

def main():
    args = parse_args()
    config = load_config(args.config)
    if args.use_gradnorm is not None:
        config['use_gradnorm'] = args.use_gradnorm
    if args.freeze_stage1 is not None:
        config['freeze_stage1'] = args.freeze_stage1

    # --- Distributed setup ---
    dist_mode = config.get('distributed', 'false')
    if args.distributed is not None:
        dist_mode = args.distributed
    is_dist = dist_mode in ('ddp', 'fsdp') and 'RANK' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if is_dist:
        setup_dist()
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(config.get('seed_value', 42))
    if is_main_process():
        logger.info(f"Using device: {device} | distributed: {dist_mode}")

    # 1. Dataset & DataLoader
    raw_dir = config['data']['raw_dir']
    csv_path = os.path.join(raw_dir, 'train', 'train.csv')
    img_dir = os.path.join(raw_dir, 'train_images')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    full_dataset = PetFinderDataset(csv_path, img_dir, transform=get_transforms(train=True))
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = get_transforms(train=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_dist else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            sampler=val_sampler, shuffle=False,
                            num_workers=num_workers)

    # 2. Model
    model = TwoStageResNet(extract_features=False).to(device)

    # --checkpoint: load weights and jump straight to Grad-CAM
    if args.checkpoint is not None:
        if is_main_process():
            logger.info(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        if is_main_process():
            config_stem = os.path.splitext(os.path.basename(args.config))[0]
            gradcam_dir = os.path.join("results", "gradcam", config_stem)
            save_gradcam_samples(model, val_loader, device, gradcam_dir,
                                 n_samples=config.get("gradcam_samples", 16),
                                 target_key="AdoptionSpeed")
        cleanup_dist()
        return

    # 3. Optim & Loss — create before DDP wrap; parameter identity is preserved
    lr = float(config.get('learning_rate', 3e-4))
    optimizer_stage1 = AdamW(model.parameters(), lr=lr)
    optimizer_stage2 = AdamW(model.head_adoption_speed.parameters(), lr=lr)

    # Wrap with DDP / FSDP
    if dist_mode == 'fsdp' and is_dist:
        from src.utils.dist import setup_fsdp
        model = setup_fsdp(model, device, config)
    elif is_dist:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    find_unused_parameters=True)

    criterion_adoption = nn.CrossEntropyLoss()

    epochs = config.get('max_epochs', 10)

    TABULAR_KEYS = ['Type', 'FurLength', 'MaturitySize', 'Breed1', 'Health',
                    'Vaccinated', 'Dewormed', 'Sterilized', 'Gender', 'Color1']
    criteria = {k: nn.CrossEntropyLoss() for k in TABULAR_KEYS}

    # ------------------ STAGE 1: Train Tabular Heads ------------------
    use_gradnorm = config.get('use_gradnorm', True)
    if use_gradnorm:
        gradnorm = GradNormController(
            n_tasks=len(TABULAR_KEYS),
            alpha=float(config.get('gradnorm_alpha', 1.5)),
            device=str(device),
        )
        gradnorm.set_weight_optimizer(lr=float(config.get('gradnorm_lr', 1e-2)))
        # Last BN of the shared backbone — representative shared layer for gradient norm computation
        shared_param = _unwrap(model).backbone.layer4[-1].bn2.weight

    stage1_label = "GradNorm" if use_gradnorm else "Equal Weights"
    if is_main_process():
        logger.info(f"=== STAGE 1: Training Tabular Heads ({stage1_label}) ===")
    global_epoch = 0
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(global_epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{epochs} [Train]",
                    disable=not is_main_process())

        for images, targets in pbar:
            images = images.to(device)
            outputs = model(images, stage=1)

            if use_gradnorm:
                losses = torch.stack([criteria[k](outputs[k], targets[k].to(device)) for k in TABULAR_KEYS])
                info = gradnorm.step(losses, shared_param, optimizer_stage1)
                pbar.set_postfix({'loss': f"{info['total_loss']:.4f}", 'gn': f"{info['gradnorm_loss']:.4f}"})
            else:
                optimizer_stage1.zero_grad()
                loss = sum(criteria[k](outputs[k], targets[k].to(device)) for k in TABULAR_KEYS)
                loss.backward()
                optimizer_stage1.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        correct = {k: 0 for k in TABULAR_KEYS}
        s1_preds = {k: [] for k in TABULAR_KEYS}
        s1_targets = {k: [] for k in TABULAR_KEYS}
        total = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Stage 1 - Epoch {epoch+1}/{epochs} [Val]",
                                        disable=not is_main_process()):
                images = images.to(device)
                outputs = model(images, stage=1)
                total += images.size(0)
                for k in TABULAR_KEYS:
                    _, pred = torch.max(outputs[k], 1)
                    correct[k] += (pred == targets[k].to(device)).sum().item()
                    s1_preds[k].extend(pred.cpu().numpy())
                    s1_targets[k].extend(targets[k].numpy())

        # Gather per-head predictions from all ranks before computing metrics
        for k in TABULAR_KEYS:
            s1_preds[k] = _gather_list(s1_preds[k], is_dist)
            s1_targets[k] = _gather_list(s1_targets[k], is_dist)
        if is_dist:
            correct_t = torch.tensor([correct[k] for k in TABULAR_KEYS], dtype=torch.long, device=device)
            total_t = torch.tensor(total, dtype=torch.long, device=device)
            tdist.all_reduce(correct_t, op=tdist.ReduceOp.SUM)
            tdist.all_reduce(total_t, op=tdist.ReduceOp.SUM)
            correct = {k: correct_t[i].item() for i, k in enumerate(TABULAR_KEYS)}
            total = total_t.item()

        accs = {k: 100 * correct[k] / total for k in TABULAR_KEYS}
        qwks_s1 = {k: cohen_kappa_score(s1_targets[k], s1_preds[k], weights='quadratic') for k in TABULAR_KEYS}
        if is_main_process():
            logger.info(
                f"Stage 1 Epoch {epoch+1} Val — " +
                " | ".join(f"{k}: acc={accs[k]:.1f}% qwk={qwks_s1[k]:.3f}" for k in TABULAR_KEYS)
            )
            if use_gradnorm:
                weights_str = " | ".join(f"{k}:{w:.3f}" for k, w in zip(TABULAR_KEYS, gradnorm.current_weights))
                logger.info(f"Stage 1 Epoch {epoch+1} GradNorm Weights — {weights_str}")
        global_epoch += 1

    if is_main_process():
        _save_checkpoint(model, config, stage=1, args=args)

    # ------------------ STAGE 2: Train AdoptionSpeed Head ------------------
    freeze_s1 = config.get('freeze_stage1', True)
    if freeze_s1:
        _unwrap(model).freeze_stage1()
        optimizer_s2 = optimizer_stage2  # head_adoption_speed only
    else:
        optimizer_s2 = AdamW(model.parameters(), lr=lr)  # end-to-end fine-tune

    stage2_label = "Frozen Stage 1" if freeze_s1 else "End-to-End"
    if is_main_process():
        logger.info(f"=== STAGE 2: Training AdoptionSpeed Head ({stage2_label}) ===")

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(global_epoch)
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{epochs} [Train]",
                    disable=not is_main_process())

        for images, targets in pbar:
            images = images.to(device)
            target_adoption = targets['AdoptionSpeed'].to(device)

            optimizer_s2.zero_grad()
            outputs = model(images)
            
            loss = criterion_adoption(outputs['AdoptionSpeed'], target_adoption)
            loss.backward()
            optimizer_s2.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Stage 2 - Epoch {epoch+1}/{epochs} [Val]",
                                        disable=not is_main_process()):
                images = images.to(device)
                target_adoption = targets['AdoptionSpeed'].to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs['AdoptionSpeed'], 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target_adoption.cpu().numpy())
                
        all_preds = _gather_list(all_preds, is_dist)
        all_targets = _gather_list(all_targets, is_dist)

        val_acc = 100 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        val_qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
        if is_main_process():
            logger.info(f"Epoch {epoch+1} - Val AdoptionSpeed — Acc: {val_acc:.2f}% | QWK: {val_qwk:.4f}\n")
        global_epoch += 1

    if is_main_process():
        _save_checkpoint(model, config, stage=2, args=args)

    # --- Classification Report (rank 0 only) ---
    if is_main_process():
        ADOPTION_CLASSES = ["Speed 0", "Speed 1", "Speed 2", "Speed 3", "Speed 4"]
        report = {
            "exp_name": config.get("exp_name", ""),
            "stage1_final_val_accuracies": {k: round(v, 2) for k, v in accs.items()},
            "stage1_final_val_qwks": {k: round(v, 4) for k, v in qwks_s1.items()},
            **({"stage1_final_task_weights": {k: round(w, 4) for k, w in zip(TABULAR_KEYS, gradnorm.current_weights)}} if use_gradnorm else {}),
            "stage2": {
                **build_clf_section(all_targets, all_preds, ADOPTION_CLASSES),
                "accuracy": round(val_acc, 2),
                "qwk": round(val_qwk, 4),
            },
            "breed1": build_breed_section(s1_targets['Breed1'], s1_preds['Breed1']),
        }
        report_path = save_report(args.config, report)
        logger.info(f"Report saved to {report_path}")

        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        gradcam_dir = os.path.join("results", "gradcam", config_stem)
        n_samples = config.get("gradcam_samples", 16)
        save_gradcam_samples(
            _unwrap(model), val_loader, device, gradcam_dir,
            n_samples=n_samples, target_key="AdoptionSpeed",
        )

    cleanup_dist()

if __name__ == "__main__":
    main()
