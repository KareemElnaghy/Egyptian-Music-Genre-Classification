"""
This file defines the EgyptianMusicDataset class for loading audio clips and their metadata,
as well as a helper function for creating song-level stratified k-fold splits.
The dataset returns tuples of (waveform_tensor, feature_tensor, label_int) for each clip.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

from augmentation import AudioAugmentation
from features import HandcraftedFeatureExtractor, FEATURE_DIM


CLASSES = ["Tarab", "Egyptian Pop", "Mahraganat", "Shaabi", "Egyptian Rap"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)} # create maps from genre name to integer label


# ─── Dataset ──────────────────────────────────────────────────────────────────

class EgyptianMusicDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        sample_rate: int = 32000,
        audio_duration: int = 30,
        augment: bool = False,
        augmentation_cfg: dict = None,
        feature_extractor: Optional[HandcraftedFeatureExtractor] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length = sample_rate * audio_duration
        self.augment = augment
        self.extractor = feature_extractor

        self.augmentor = None
        if augment and augmentation_cfg is not None:
            self.augmentor = AudioAugmentation(augmentation_cfg, sample_rate)

    def set_epoch(self, epoch: int):
        """Propagate current epoch to augmentor for warmup scheduling."""
        if self.augmentor is not None:
            self.augmentor.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        genre: str = row["genre"]
        filename: str = row["filename"]

        audio_path = os.path.join(self.audio_dir, genre, filename)
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Trim or zero-pad to fixed length
        if len(waveform) >= self.target_length:
            waveform = waveform[: self.target_length]
        else:
            waveform = np.pad(waveform, (0, self.target_length - len(waveform)))

        if self.augment and self.augmentor is not None:
            waveform = self.augmentor(waveform)

        # Hand-crafted features (always float32, zeros when not using fusion)
        if self.extractor is not None:
            features = self.extractor.transform(waveform, sr=self.sample_rate)
        else:
            features = np.zeros(FEATURE_DIM, dtype=np.float32)

        label = CLASS_TO_IDX[genre]
        return (
            torch.from_numpy(waveform).float(),
            torch.from_numpy(features).float(),
            label,
        )


# stratified k fold

def get_song_level_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits the dataset at the song (source_file) level so that clips from
    the same recording never appear in both train and validation.
    """
    songs = (
        df.groupby("source_file")["genre"]
        .first()
        .reset_index()
    )
    songs["label"] = songs["genre"].map(CLASS_TO_IDX)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, val_idx in skf.split(songs["source_file"], songs["label"]):
        train_songs = songs.iloc[train_idx]["source_file"].values
        val_songs = songs.iloc[val_idx]["source_file"].values

        train_df = df[df["source_file"].isin(train_songs)].copy()
        val_df = df[df["source_file"].isin(val_songs)].copy()

        folds.append((train_df, val_df))

    return folds


# helper functions to load metadata and print fold summaries
def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise column names to what the rest of the code expects
    col_map = {
        "clip_filename":   "filename",
        "label":           "genre",
        "source_filename": "source_file",
        "start_time":      "start_sec",
        "end_time":        "end_sec",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def print_fold_summary(folds: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    for i, (train_df, val_df) in enumerate(folds):
        n_train_songs = train_df["source_file"].nunique()
        n_val_songs = val_df["source_file"].nunique()
        print(
            f"Fold {i+1}: "
            f"train={len(train_df)} clips / {n_train_songs} songs | "
            f"val={len(val_df)} clips / {n_val_songs} songs"
        )
