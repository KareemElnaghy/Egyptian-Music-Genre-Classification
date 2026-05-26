"""
Grad-CAM explanations targeting conv_block6 of the CNN14 backbone.
Simple explaination as to how it works:
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.ndimage import zoom


# Grad CAM implementation adapted for audio classification with PANNs CNN14 backbone.

class GradCAM:
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Call when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(
        self,
        waveform: torch.Tensor,
        class_idx: int,
        handcrafted_features: torch.Tensor = None,
    ) -> np.ndarray:
        
        device = next(self.model.parameters()).device

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device)

        if handcrafted_features is not None:
            if handcrafted_features.dim() == 1:
                handcrafted_features = handcrafted_features.unsqueeze(0)
            handcrafted_features = handcrafted_features.to(device)

        # Need gradients to flow — use model.eval() but keep grad enabled
        self.model.eval()
        self.model.zero_grad()

        with torch.enable_grad():
            output_dict = self.model(waveform, handcrafted_features)
            log_probs = output_dict["clipwise_output"]   # (1, C)
            score = log_probs[0, class_idx]
            score.backward()

        # Channel importance = GAP of gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)
        cam = (weights * self._activations).sum(dim=1)         # (1, T, F)
        cam = F.relu(cam).squeeze(0).cpu().numpy()             # (T, F)

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam   # low-res saliency map; caller upsamples to mel dims

    def __call__(self, waveform, class_idx, handcrafted_features=None):
        return self.generate(waveform, class_idx, handcrafted_features)


# Visualization

def visualise_gradcam(
    cam: np.ndarray,
    waveform: torch.Tensor,
    out_path: str,
    sample_rate: int = 32000,
    hop_length: int = 320,
    n_mels: int = 64,
    fmax: int = 14000,
    class_name: str = "",
    true_class: str = "",
    pred_class: str = "",
):
    """
    Displays two panels one of the log-mel spectrogram and the other is the Grad-CAM saliency heatmap overlaid on the log-mel.
    """
    y = waveform.squeeze().cpu().numpy().astype(np.float32)

    # Compute log-mel at the PANNs resolution
    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_mels=n_mels,
        n_fft=1024, hop_length=hop_length, fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    scale_t = log_mel.shape[1] / cam.shape[0]
    scale_f = log_mel.shape[0] / cam.shape[1]
    cam_up = zoom(cam, (scale_t, scale_f), order=1)
    cam_up = cam_up.T

    title_suffix = ""
    if true_class:
        title_suffix += f"  true: {true_class}"
    if pred_class:
        correct = "✓" if true_class == pred_class else "✗"
        title_suffix += f"  pred: {pred_class} {correct}"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # log-mel spectrogram panel
    img1 = librosa.display.specshow(
        log_mel, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="mel", fmax=fmax, ax=axes[0],
    )
    axes[0].set_title(f"Log-mel spectrogram{title_suffix}", fontsize=11)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    # Grad-CAM panel
    axes[1].imshow(
        log_mel, aspect="auto", origin="lower", cmap="gray",
        extent=[0, len(y) / sample_rate, 0, n_mels],
        alpha=0.6,
    )
    heat = axes[1].imshow(
        cam_up, aspect="auto", origin="lower", cmap="hot",
        extent=[0, len(y) / sample_rate, 0, n_mels],
        alpha=0.6,
    )
    axes[1].set_title("Grad-CAM saliency (conv_block6)", fontsize=11)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(heat, ax=axes[1], label="Saliency")

    plt.suptitle(class_name or "Grad-CAM", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM saved → {out_path}")
