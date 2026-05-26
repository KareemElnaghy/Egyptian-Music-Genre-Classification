"""
Training loop for Egyptian music genre classification.
"""

import json
import os
import csv
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EgyptianMusicDataset, get_song_level_folds, load_metadata, CLASS_TO_IDX
from features import HandcraftedFeatureExtractor
from model import build_model

# set seed so that it is reproducible
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Split dataset

def get_fixed_split(df, split_path: str, seed: int = 42):
    """
    Song-level stratified 70% train / 15% val / 15% test split.
    """
    from sklearn.model_selection import train_test_split

    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
    else:
        songs = (
            df.groupby("source_file")["genre"]
            .first()
            .reset_index()
        )
        songs["label"] = songs["genre"].map(CLASS_TO_IDX)

        train_songs, temp = train_test_split(
            songs, test_size=0.30, stratify=songs["label"], random_state=seed
        )
        val_songs, test_songs = train_test_split(
            temp, test_size=0.50, stratify=temp["label"], random_state=seed
        )

        split = {
            "train_songs": train_songs["source_file"].tolist(),
            "val_songs":   val_songs["source_file"].tolist(),
            "test_songs":  test_songs["source_file"].tolist(),
        }
        os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"  Split saved → {split_path}  "
              f"(train {len(split['train_songs'])} / "
              f"val {len(split['val_songs'])} / "
              f"test {len(split['test_songs'])} songs)")

    train_df = df[df["source_file"].isin(split["train_songs"])].copy()
    val_df = df[df["source_file"].isin(split["val_songs"])].copy()
    test_df = df[df["source_file"].isin(split["test_songs"])].copy()
    return train_df, val_df, test_df


# helpers for training loop i.e. loss functions, mixup, and epoch runner

def smooth_nll_loss(log_probs: torch.Tensor, labels: torch.Tensor,
                    num_classes: int, eps: float = 0.1) -> torch.Tensor:

    with torch.no_grad():
        smooth = torch.full_like(log_probs, eps / (num_classes - 1))
        smooth.scatter_(1, labels.unsqueeze(1), 1.0 - eps)
    return F.kl_div(log_probs, smooth, reduction="batchmean")


def soft_nll_loss(log_probs: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    return F.kl_div(log_probs, soft_labels, reduction="batchmean")


def make_criterion(cfg: dict, num_classes: int):
    use_smoothing = cfg.get("use_label_smoothing", False)
    eps = cfg.get("label_smoothing_eps", 0.1)

    def criterion(log_probs, labels):
        if labels.dtype == torch.float32:
            return soft_nll_loss(log_probs, labels)
        if use_smoothing:
            return smooth_nll_loss(log_probs, labels, num_classes, eps)
        return F.nll_loss(log_probs, labels)

    return criterion


def mixup_batch(waveforms: torch.Tensor, features: torch.Tensor,
                labels: torch.Tensor, num_classes: int, alpha: float = 0.3):

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(waveforms.size(0), device=waveforms.device)

    # One-hot encode integer labels → (B, C) float
    one_hot = torch.zeros(labels.size(0), num_classes,
                          dtype=torch.float32, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    w_mix = lam * waveforms + (1 - lam) * waveforms[idx]
    f_mix = lam * features  + (1 - lam) * features[idx]
    y_mix = lam * one_hot   + (1 - lam) * one_hot[idx]

    return w_mix, f_mix, y_mix


# single epoch runner

def run_epoch(model, loader, optimizer, criterion, device,
              is_train: bool, desc: str = "",
              use_mixup: bool = False, mixup_alpha: float = 0.3,
              num_classes: int = 5):
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(loader, desc=desc, leave=False, ncols=90, unit="batch")
    with torch.set_grad_enabled(is_train):
        for waveforms, features, labels in bar:
            waveforms = waveforms.to(device)
            features = features.to(device)
            labels = labels.to(device)

            if is_train and use_mixup:
                waveforms, features, labels = mixup_batch(
                    waveforms, features, labels, num_classes, mixup_alpha
                )

            output_dict = model(waveforms, features)
            log_probs = output_dict["clipwise_output"]

            loss = criterion(log_probs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * (waveforms.size(0))
            # For accuracy: use hard pred from log_probs, true label from argmax if soft
            preds = log_probs.argmax(dim=1)
            if labels.dtype == torch.float32:
                hard_labels = labels.argmax(dim=1)
            else:
                hard_labels = labels
            correct += (preds == hard_labels).sum().item()
            total   += waveforms.size(0)

            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


# our optimizer

def build_optimizer_and_scheduler(model, cfg: dict, epochs_remaining: int):
    """Build AdamW + CosineAnnealingLR from cfg, using model.get_parameter_groups()."""
    param_groups = model.get_parameter_groups(
        base_lr=cfg["learning_rate"],
        backbone_lr_multiplier=cfg.get("backbone_lr_multiplier", 0.1),
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_remaining, eta_min=1e-6
    )
    return optimizer, scheduler


# k fold cv training
def train_kfold(cfg: dict, run_dir: str) -> str:
    set_seed(cfg["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    df = load_metadata(cfg["data_csv"])
    folds = get_song_level_folds(df, n_splits=cfg["num_folds"], seed=cfg["seed"])

    augment_cfg = cfg.get("augmentation", {})
    use_fusion = cfg.get("use_feature_fusion", False)
    num_classes = len(cfg["classes"])

    fold_val_accs = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        fold_num = fold_idx + 1
        fold_dir = os.path.join(run_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_num}/{cfg['num_folds']}  –  "
              f"train {len(train_df)} clips | val {len(val_df)} clips")
        print(f"{'='*60}")

        extractor = None
        if use_fusion:
            extractor = HandcraftedFeatureExtractor()
            extractor.fit_on_df(train_df, cfg["audio_dir"], cfg["sample_rate"],
                                cfg["audio_duration"])
            extractor.save(os.path.join(fold_dir, "feature_scaler.pkl"))

        train_ds = EgyptianMusicDataset(
            train_df, cfg["audio_dir"],
            sample_rate=cfg["sample_rate"],
            audio_duration=cfg["audio_duration"],
            augment=cfg.get("use_augmentation", False),
            augmentation_cfg=augment_cfg,
            feature_extractor=extractor,
        )
        val_ds = EgyptianMusicDataset(
            val_df, cfg["audio_dir"],
            sample_rate=cfg["sample_rate"],
            audio_duration=cfg["audio_duration"],
            augment=False,
            feature_extractor=extractor,
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
        )

        model = build_model(cfg).to(device)
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, cfg["epochs"])
        criterion = make_criterion(cfg, num_classes)

        best_val_acc = 0.0
        best_ckpt_path = os.path.join(fold_dir, "best.pth")
        log_path = os.path.join(fold_dir, "train_log.csv")

        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
            )

        epoch_bar = tqdm(range(1, cfg["epochs"] + 1),
                         desc=f"Fold {fold_num}", ncols=90, unit="epoch")
        for epoch in epoch_bar:
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)

            train_loss, train_acc = run_epoch(
                model, train_loader, optimizer, criterion, device,
                is_train=True, desc="  train",
                use_mixup=False, num_classes=num_classes,
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, optimizer, criterion, device,
                is_train=False, desc="  val  ",
            )
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]

            epoch_bar.set_postfix(
                tr_loss=f"{train_loss:.4f}", tr_acc=f"{train_acc:.4f}",
                vl_loss=f"{val_loss:.4f}",  vl_acc=f"{val_acc:.4f}",
            )

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, train_loss, train_acc, val_loss, val_acc, lr_now]
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "val_acc": val_acc, "cfg": cfg},
                    best_ckpt_path,
                )
                print(f"  ✓ New best val acc {best_val_acc:.4f} — checkpoint saved")

        fold_val_accs.append(best_val_acc)
        print(f"\n  Fold {fold_num} best val acc: {best_val_acc:.4f}")

    mean_acc = float(np.mean(fold_val_accs))
    std_acc = float(np.std(fold_val_accs))
    print(f"\n{'='*60}")
    print(f"  5-Fold CV complete")
    print(f"  Val acc per fold: {[f'{a:.4f}' for a in fold_val_accs]}")
    print(f"  Mean ± std: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Results → {run_dir}")
    print(f"{'='*60}\n")

    with open(os.path.join(run_dir, "cv_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_val_acc"])
        for i, acc in enumerate(fold_val_accs, 1):
            w.writerow([i, acc])
        w.writerow(["mean", mean_acc])
        w.writerow(["std",  std_acc])

    return run_dir


# single fixed-split training (70/15/15)

def train_single_split(cfg: dict, run_dir: str) -> str:
    set_seed(cfg["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    df = load_metadata(cfg["data_csv"])

    split_path = cfg.get("split_path", "data/phase4_split.json")
    if not os.path.isabs(split_path):
        split_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            split_path,
        )

    train_df, val_df, test_df = get_fixed_split(df, split_path, seed=cfg["seed"])
    print(f"\n  Split: train {len(train_df)} clips | "
          f"val {len(val_df)} clips | test {len(test_df)} clips")

    augment_cfg = cfg.get("augmentation", {})
    use_fusion = cfg.get("use_feature_fusion", False)
    use_mixup = cfg.get("use_mixup", False)
    mixup_alpha = cfg.get("mixup_alpha", 0.3)
    num_classes = len(cfg["classes"])

    extractor = None
    if use_fusion:
        extractor = HandcraftedFeatureExtractor()
        extractor.fit_on_df(train_df, cfg["audio_dir"], cfg["sample_rate"],
                            cfg["audio_duration"])
        extractor.save(os.path.join(run_dir, "feature_scaler.pkl"))

    train_ds = EgyptianMusicDataset(
        train_df, cfg["audio_dir"],
        sample_rate=cfg["sample_rate"],
        audio_duration=cfg["audio_duration"],
        augment=cfg.get("use_augmentation", False),
        augmentation_cfg=augment_cfg,
        feature_extractor=extractor,
    )
    val_ds = EgyptianMusicDataset(
        val_df, cfg["audio_dir"],
        sample_rate=cfg["sample_rate"],
        audio_duration=cfg["audio_duration"],
        augment=False,
        feature_extractor=extractor,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    model = build_model(cfg).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, cfg["epochs"])
    criterion = make_criterion(cfg, num_classes)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(run_dir, "best.pth")
    log_path = os.path.join(run_dir, "train_log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )

    epoch_bar = tqdm(range(1, cfg["epochs"] + 1), desc="Training", ncols=90, unit="epoch")
    for epoch in epoch_bar:
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device,
            is_train=True, desc="  train",
            use_mixup=use_mixup, mixup_alpha=mixup_alpha, num_classes=num_classes,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, device,
            is_train=False, desc="  val  ",
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}", tr_acc=f"{train_acc:.4f}",
            vl_loss=f"{val_loss:.4f}",  vl_acc=f"{val_acc:.4f}",
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc, lr_now]
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_acc": val_acc, "cfg": cfg},
                best_ckpt_path,
            )
            print(f"  ✓ New best val acc {best_val_acc:.4f} — checkpoint saved")

    print(f"\n  Training complete.  Best val acc: {best_val_acc:.4f}")
    print(f"  Results → {run_dir}")

    with open(os.path.join(run_dir, "cv_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_val_acc"])
        w.writerow(["1", best_val_acc])
        w.writerow(["mean", best_val_acc])
        w.writerow(["std", 0.0])

    with open(os.path.join(run_dir, "test_songs.json"), "w") as f:
        json.dump(test_df["source_file"].unique().tolist(), f, indent=2)

    return run_dir


# train entry point

def train(cfg: dict) -> str:
    """Dispatch to single-split or k-fold training based on use_single_split."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = cfg.get("experiment_name", timestamp)
    run_dir = os.path.join(cfg["results_dir"], folder_name)
    os.makedirs(run_dir, exist_ok=True)

    if cfg.get("use_single_split", False):
        return train_single_split(cfg, run_dir)
    return train_kfold(cfg, run_dir)
