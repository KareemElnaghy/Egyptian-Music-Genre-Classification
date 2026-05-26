"""
Hand-crafted acoustic feature extraction
Here we do the feature extraction mention in our model 
the types of features extracted are chroma, mfcc, spectral contrast and zero crossing rate
"""

import os
import pickle

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

FEATURE_DIM = 72


# low level extractors
def extract_raw(waveform: np.ndarray, sr: int = 32000) -> np.ndarray:
    """
    Extract a 72-dim feature vector from a raw waveform (unscaled).
    """
    hop = 512
    n_fft = 1024

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr, n_chroma=12,
                                          n_fft=n_fft, hop_length=hop)
    chroma_feats = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])

    # MFCCs
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20,
                                  n_fft=n_fft, hop_length=hop)
    mfcc_feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_bands=6,
                                                   n_fft=n_fft, hop_length=hop)
    contrast_feats = contrast.mean(axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=waveform, hop_length=hop)
    zcr_feats = np.array([zcr.mean()])

    features = np.concatenate([chroma_feats, mfcc_feats, contrast_feats, zcr_feats])
    assert features.shape == (FEATURE_DIM,), f"Expected {FEATURE_DIM} dims, got {features.shape}"
    return features.astype(np.float32)


# extractor class

class HandcraftedFeatureExtractor:
    """
    This class encapsulates the extraction and scaling of hand-crafted features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    # fitting
    def fit_on_df(self, df, audio_dir: str, sample_rate: int = 32000,
                  audio_duration: int = 30):
        """
        Load every clip in df, extract raw features, and fit the scaler.

        Logs progress every 50 clips.
        """
        target_len = sample_rate * audio_duration
        raw_matrix = []

        print(f"    Fitting feature scaler on {len(df)} clips …")
        for i, (_, row) in enumerate(df.iterrows()):
            path = os.path.join(audio_dir, row["genre"], row["filename"])
            y, _ = librosa.load(path, sr=sample_rate, mono=True)
            if len(y) >= target_len:
                y = y[:target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)))
            raw_matrix.append(extract_raw(y, sr=sample_rate))
            if (i + 1) % 50 == 0:
                print(f"    … {i+1}/{len(df)} clips processed")

        self.scaler.fit(np.stack(raw_matrix))
        self._fitted = True
        print(f"    Scaler fitted on {len(df)} clips.")

    # transformation
    def transform(self, waveform: np.ndarray, sr: int = 32000) -> np.ndarray:
        """
        Extract features and normalise with the fitted scaler.

        Returns float32 numpy array of shape (FEATURE_DIM,).
        Falls back to zero-mean unit-var-scaled zeros if not fitted.
        """
        raw = extract_raw(waveform, sr=sr).reshape(1, -1)
        if self._fitted:
            return self.scaler.transform(raw).squeeze(0).astype(np.float32)
        return raw.squeeze(0)

    # persistence
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> "HandcraftedFeatureExtractor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        extractor = cls()
        extractor.scaler = data["scaler"]
        extractor._fitted = data["fitted"]
        return extractor
