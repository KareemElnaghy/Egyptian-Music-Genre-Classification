"""
Post-training evaluation.
This module loads the best checkpoint from each fold, runs inference on the fold's validation set,
aggregates predictions across folds, and computes all metrics.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from scipy.special import softmax as scipy_softmax

from dataset import (
    EgyptianMusicDataset,
    CLASSES,
    get_song_level_folds,
    load_metadata,
)
from features import HandcraftedFeatureExtractor
from model import build_model



@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_log_probs = []

    for waveforms, features, labels in loader:
        waveforms = waveforms.to(device)
        features = features.to(device)
        out = model(waveforms, features)
        all_log_probs.append(out["clipwise_output"].cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_log_probs)


# helper functions to compute per-class accuracy and save confusion matrix visualisations
def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    accs = {}
    for i, cls in enumerate(CLASSES[:num_classes]):
        mask = y_true == i
        if mask.sum() == 0:
            accs[cls] = float("nan")
        else:
            accs[cls] = (y_pred[mask] == i).mean()
    return accs


def save_confusion_matrix(y_true, y_pred, classes, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved {out_path}")


# main evaluation function for cross-validation results
def evaluate(cfg: dict, run_dir: str) -> dict:
    """
    Load the best checkpoint for each fold, run inference on the fold's
    validation set, aggregate predictions, and compute all metrics.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )

    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    df = load_metadata(cfg["data_csv"])
    folds = get_song_level_folds(df, n_splits=cfg["num_folds"], seed=cfg["seed"])

    num_classes = len(cfg["classes"])

    all_labels = []
    all_log_probs = []

    for fold_idx, (_, val_df) in enumerate(folds):
        fold_num = fold_idx + 1
        ckpt_path = os.path.join(run_dir, f"fold_{fold_num}", "best.pth")

        if not os.path.exists(ckpt_path):
            print(f"  [WARNING] Checkpoint not found for fold {fold_num}, skipping.")
            continue

        print(f"\n  Evaluating fold {fold_num} …")

        # Load model
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Load per-fold feature scaler if fusion was used
        extractor = None
        if cfg.get("use_feature_fusion", False):
            scaler_path = os.path.join(run_dir, f"fold_{fold_num}", "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                extractor = HandcraftedFeatureExtractor.load(scaler_path)
            else:
                print(f"  [WARNING] Feature scaler not found at {scaler_path}")

        val_ds = EgyptianMusicDataset(
            val_df, cfg["audio_dir"],
            sample_rate=cfg["sample_rate"],
            audio_duration=cfg["audio_duration"],
            augment=False,
            feature_extractor=extractor,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )

        labels, log_probs = collect_predictions(model, val_loader, device)
        all_labels.append(labels)
        all_log_probs.append(log_probs)

    all_labels = np.concatenate(all_labels)
    all_log_probs = np.concatenate(all_log_probs)
    all_probs = scipy_softmax(all_log_probs, axis=1)   # for ROC-AUC
    all_preds = all_log_probs.argmax(axis=1)

    # metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cls_accs = per_class_accuracy(all_labels, all_preds, num_classes)

    # ROC-AUC 
    try:
        roc_auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro",
        )
    except ValueError as exc:
        roc_auc = float("nan")

    report = classification_report(
        all_labels, all_preds,
        target_names=cfg["classes"],
        zero_division=0,
    )

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS  (aggregated over {cfg['num_folds']} folds)")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {acc:.4f}")
    print(f"  Macro F1 score   : {macro_f1:.4f}")
    print(f"  ROC-AUC (OvR)   : {roc_auc:.4f}")
    print(f"\n  Per-class accuracy:")
    for cls, a in cls_accs.items():
        print(f"    {cls:<20s}: {a:.4f}")
    print(f"\n{report}")

    save_confusion_matrix(
        all_labels, all_preds,
        classes=cfg["classes"],
        out_path=os.path.join(eval_dir, "confusion_matrix.png"),
    )

    # Numerical summary CSV
    summary_rows = [
        ["metric", "value"],
        ["accuracy",  acc],
        ["macro_f1",  macro_f1],
        ["roc_auc",   roc_auc],
    ]
    for cls, a in cls_accs.items():
        summary_rows.append([f"acc_{cls}", a])

    summary_df = pd.DataFrame(summary_rows[1:], columns=summary_rows[0])
    summary_df.to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

    # Classification report text
    with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    return {
        "accuracy":              acc,
        "macro_f1":              macro_f1,
        "roc_auc":               roc_auc,
        "per_class_acc":         cls_accs,
        "classification_report": report,
    }


def evaluate_single_split(cfg: dict, run_dir: str) -> dict:

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    test_dir = os.path.join(run_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    ckpt_path = os.path.join(run_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Run training first."
        )

    # Reconstruct test DataFrame from saved song list
    test_songs_path = os.path.join(run_dir, "test_songs.json")
    if not os.path.exists(test_songs_path):
        raise FileNotFoundError(
            f"test_songs.json not found at {run_dir}. "
            "Was this run trained with use_single_split: true?"
        )

    df = load_metadata(cfg["data_csv"])
    with open(test_songs_path) as f:
        test_songs = json.load(f)
    test_df = df[df["source_file"].isin(test_songs)].copy()

    print(f"\n  Test set: {len(test_df)} clips from {test_df['source_file'].nunique()} songs")

    # Load feature scaler if fusion was used
    extractor = None
    if cfg.get("use_feature_fusion", False):
        scaler_path = os.path.join(run_dir, "feature_scaler.pkl")
        if os.path.exists(scaler_path):
            extractor = HandcraftedFeatureExtractor.load(scaler_path)
        else:
            print(f"  [WARNING] Feature scaler not found at {scaler_path}")

    test_ds = EgyptianMusicDataset(
        test_df, cfg["audio_dir"],
        sample_rate=cfg["sample_rate"],
        audio_duration=cfg["audio_duration"],
        augment=False,
        feature_extractor=extractor,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    all_labels, all_log_probs = collect_predictions(model, test_loader, device)
    all_probs = scipy_softmax(all_log_probs, axis=1)
    all_preds = all_log_probs.argmax(axis=1)

    num_classes = len(cfg["classes"])
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cls_accs = per_class_accuracy(all_labels, all_preds, num_classes)

    try:
        roc_auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError as exc:
        roc_auc = float("nan")
        print(f"  [WARNING] ROC-AUC skipped: {exc}")

    report = classification_report(
        all_labels, all_preds, target_names=cfg["classes"], zero_division=0
    )

    print(f"\n{'='*60}")
    print(f"  TEST-SET RESULTS  (fixed split)")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {acc:.4f}")
    print(f"  Macro F1 score   : {macro_f1:.4f}")
    print(f"  ROC-AUC (OvR)   : {roc_auc:.4f}")
    print(f"\n  Per-class accuracy:")
    for cls, a in cls_accs.items():
        print(f"    {cls:<20s}: {a:.4f}")
    print(f"\n{report}")

    save_confusion_matrix(
        all_labels, all_preds,
        classes=cfg["classes"],
        out_path=os.path.join(test_dir, "confusion_matrix.png"),
    )

    summary_rows = [
        ["metric", "value"],
        ["accuracy",  acc],
        ["macro_f1",  macro_f1],
        ["roc_auc",   roc_auc],
    ]
    for cls, a in cls_accs.items():
        summary_rows.append([f"acc_{cls}", a])

    summary_df = pd.DataFrame(summary_rows[1:], columns=summary_rows[0])
    summary_df.to_csv(os.path.join(test_dir, "metrics.csv"), index=False)

    with open(os.path.join(test_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(f"  Test-set outputs saved → {test_dir}")

    return {
        "accuracy":              acc,
        "macro_f1":              macro_f1,
        "roc_auc":               roc_auc,
        "per_class_acc":         cls_accs,
        "classification_report": report,
    }
