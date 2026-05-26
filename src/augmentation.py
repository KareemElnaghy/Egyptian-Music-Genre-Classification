"""
Data augmentation pipeline for Egyptian music genre classification.
"""

import hashlib
import os
import random

import numpy as np
import librosa
import torch
import torch.nn as nn


# SpecAugment implementation

class SpecAugmentModule(nn.Module):
    """
    Frequency and time masking for log-mel spectrograms
    We mask to stabilize training
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T, F)"""
        if not self.training:
            return x

        x = x.clone()
        B, C, T, F = x.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(F - f, 0))
            x[:, :, :, f0: f0 + f] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(T - t, 0))
            x[:, :, t0: t0 + t, :] = 0.0

        return x

    def extra_repr(self) -> str:
        return (f"freq_mask_param={self.freq_mask_param}, "
                f"time_mask_param={self.time_mask_param}, "
                f"num_freq_masks={self.num_freq_masks}, "
                f"num_time_masks={self.num_time_masks}")


# Audio Augmentation Pipeline

class AudioAugmentation:
    """
    Config keys all from the config.yaml file
    """

    def __init__(self, cfg: dict, sample_rate: int = 32000):
        self.sr = sample_rate
        self.cfg = cfg

        self.apply_prob = cfg.get("aug_apply_prob", 1.0)

        # Pitch shift
        self.use_pitch = cfg.get("use_pitch_shift", True)
        self.pitch_prob = cfg.get("pitch_shift_prob", 0.5)
        self.pitch_steps = cfg.get("pitch_shift_steps", [-2, -1, 1, 2])

        # Time stretch
        self.use_stretch = cfg.get("use_time_stretch", True)
        self.stretch_prob = cfg.get("time_stretch_prob", 0.5)
        stretch_range = cfg.get("time_stretch_range", [0.85, 1.15])
        self.stretch_min, self.stretch_max = stretch_range[0], stretch_range[1]

        # Noise
        self.use_noise = cfg.get("use_noise", True)
        self.noise_prob = cfg.get("noise_prob", 0.4)
        snr_range = cfg.get("noise_snr_db", [15, 30])
        self.snr_min, self.snr_max = snr_range[0], snr_range[1]

        # Cache
        self.use_cache = cfg.get("cache_augmented", False)
        self.cache_dir = cfg.get("cache_dir", "/tmp/egyptian_music_aug_cache")
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Warmup state - where we are in the training schedule, used to scale augmentation probabilities
        self.current_epoch = 0
        self._scale = 1.0

    # Individual augmentations

    def _pitch_shift(self, waveform: np.ndarray) -> np.ndarray:
        n_steps = float(random.choice(self.pitch_steps))
        return librosa.effects.pitch_shift(waveform, sr=self.sr, n_steps=n_steps)

    def _time_stretch(self, waveform: np.ndarray) -> np.ndarray:
        rate = random.uniform(self.stretch_min, self.stretch_max)
        stretched = librosa.effects.time_stretch(waveform, rate=rate)
        # Restore original length
        target = len(waveform)
        if len(stretched) >= target:
            return stretched[:target]
        return np.pad(stretched, (0, target - len(stretched)))

    def _add_noise(self, waveform: np.ndarray) -> np.ndarray:
        snr_db = random.uniform(self.snr_min, self.snr_max)
        signal_std = waveform.std()
        if signal_std == 0:
            return waveform
        noise_std = signal_std / (10 ** (snr_db / 20.0))
        noise = np.random.normal(0.0, noise_std, size=waveform.shape).astype(np.float32)
        return waveform + noise

    # Warmup scheduler

    def set_epoch(self, epoch: int):
        """
        Called by train.py at the start of each epoch.
        Controls warmup and linear rampup of augmentation probabilities.
        Gradual introduction of augmentation helps stabilize training, especially on a small dataset.

        In short: 
        epoch < warmup_epochs                     scale = 0.0 (no augmentation)
        warmup_epochs ≤ epoch < warmup+rampup     scale linearly 0 --> 1
        epoch ≥ warmup_epochs + rampup_epochs     scale = 1.0 (full augmentation)
        """
        self.current_epoch = epoch
        warmup = self.cfg.get("aug_warmup_epochs", 0)
        rampup = self.cfg.get("aug_rampup_epochs", 0)

        if epoch < warmup:
            self._scale = 0.0
        elif rampup == 0 or epoch >= warmup + rampup:
            self._scale = 1.0
        else:
            self._scale = (epoch - warmup) / rampup

    def _scaled_prob(self, prob: float) -> float:
        return prob * self._scale

    # Cache helper functions

    def _cache_key(self, waveform: np.ndarray) -> str:
        h = hashlib.sha256(waveform.tobytes()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{h}.npy")

    def _load_cache(self, key: str):
        if os.path.exists(key):
            return np.load(key)
        return None

    def _save_cache(self, key: str, waveform: np.ndarray):
        np.save(key, waveform)

    # Augmentation interfaces

    def augment_waveform(self, waveform: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Apply the waveform-level augmentation pipeline.
        Cache hit returns the previously-augmented version (deterministic).
        """
        if self.use_cache:
            key = self._cache_key(waveform)
            cached = self._load_cache(key)
            if cached is not None:
                return cached

        # Master probability gate (also scaled by warmup ramp)
        if random.random() > self._scaled_prob(self.apply_prob):
            return waveform

        if self.use_pitch and random.random() < self._scaled_prob(self.pitch_prob):
            waveform = self._pitch_shift(waveform)

        if self.use_stretch and random.random() < self._scaled_prob(self.stretch_prob):
            waveform = self._time_stretch(waveform)

        if self.use_noise and random.random() < self._scaled_prob(self.noise_prob):
            waveform = self._add_noise(waveform)

        if self.use_cache:
            self._save_cache(key, waveform)

        return waveform

    def augment_spectrogram(self, spec_tensor: torch.Tensor) -> torch.Tensor:
        """
        Spectrogram-level augmentation interface.
        """
        # Delegate to a temporary SpecAugmentModule in eval-free mode
        f_param = self.cfg.get("freq_mask_param", 20)
        t_param = self.cfg.get("time_mask_param", 40)
        n_freq = self.cfg.get("num_freq_masks",  2)
        n_time = self.cfg.get("num_time_masks",  2)
        module = SpecAugmentModule(f_param, t_param, n_freq, n_time).train()
        with torch.no_grad():
            return module(spec_tensor)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Backward-compatible interface used by dataset.py."""
        return self.augment_waveform(waveform, sr=self.sr)
