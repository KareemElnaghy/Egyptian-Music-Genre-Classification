"""
This file is responsible for loading the trained model and feature scaler, and providing a predict() function 
that takes an audio file path as input and outputs the predicted genre along with confidence scores.
"""

import os
import sys

import numpy as np
import torch
import librosa
from scipy.special import softmax

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_SRC = os.path.normpath(os.path.join(_FILE_DIR, "..", "..", "..", "src"))
if _ROOT_SRC not in sys.path:
    sys.path.insert(0, _ROOT_SRC)

from model import Transfer_Cnn14
from features import HandcraftedFeatureExtractor, FEATURE_DIM

CLASSES = ["Tarab", "Egyptian Pop", "Mahraganat", "Shaabi", "Egyptian Rap"]

_SAMPLE_RATE = 32000
_CLIP_DURATION = 30
_CLIP_LENGTH = _SAMPLE_RATE * _CLIP_DURATION
_MIN_REMAINDER = _SAMPLE_RATE * 5  # include tail clips of at least 5 s

_CKPT_PATH = os.path.join(_FILE_DIR, "..", "trained_model", "final_trained_model.pth")
_SCALER_PATH = os.path.join(_FILE_DIR, "..", "trained_model", "final_scaler.pkl")

_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def _build_model() -> Transfer_Cnn14:
    ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    model = Transfer_Cnn14(
        sample_rate=cfg["sample_rate"],
        window_size=cfg["window_size"],
        hop_size=cfg["hop_size"],
        mel_bins=cfg["mel_bins"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
        classes_num=len(cfg["classes"]),
        freeze_layers=cfg["freeze_layers"],
        use_attention=cfg.get("use_attention", False),
        use_feature_fusion=cfg.get("use_feature_fusion", False),
        fusion_hidden_dim=cfg.get("fusion_hidden_dim", 256),
        fusion_dropout=cfg.get("fusion_dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(_device)
    model.eval()
    return model


# Load once at module import
_model = _build_model()
_extractor = HandcraftedFeatureExtractor.load(_SCALER_PATH)


def _split_clips(waveform: np.ndarray) -> list:
    """Split a waveform into non-overlapping 30-second clips."""
    if len(waveform) <= _CLIP_LENGTH:
        return [np.pad(waveform, (0, _CLIP_LENGTH - len(waveform)))]

    clips = []
    n_full = len(waveform) // _CLIP_LENGTH
    for i in range(n_full):
        clips.append(waveform[i * _CLIP_LENGTH : (i + 1) * _CLIP_LENGTH])

    tail = waveform[n_full * _CLIP_LENGTH :]
    if len(tail) >= _MIN_REMAINDER:
        clips.append(np.pad(tail, (0, _CLIP_LENGTH - len(tail))))

    return clips


@torch.no_grad()
def predict(audio_path: str) -> dict:
    """
    Classify an audio file into one of 5 Egyptian music genres.
    """
    waveform, _ = librosa.load(audio_path, sr=_SAMPLE_RATE, mono=True)
    clips = _split_clips(waveform)

    all_log_probs = []
    for clip in clips:
        features = _extractor.transform(clip, sr=_SAMPLE_RATE)
        waveform_t = torch.from_numpy(clip).float().unsqueeze(0).to(_device)
        features_t = torch.from_numpy(features).float().unsqueeze(0).to(_device)

        out = _model(waveform_t, features_t)
        all_log_probs.append(out["clipwise_output"].cpu().numpy())

    avg_log_probs = np.mean(np.vstack(all_log_probs), axis=0)
    probs = softmax(avg_log_probs)
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_genre": CLASSES[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "scores":          {cls: float(p) for cls, p in zip(CLASSES, probs)},
        "clips_analyzed":  len(clips),
    }
