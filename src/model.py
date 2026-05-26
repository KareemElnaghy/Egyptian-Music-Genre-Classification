"""
Transfer_Cnn14: CNN14 backbone pretrained on AudioSet, adapted for Egyptian music
classification.  The backbone code is reproduced from:
  https://github.com/qiuqiangkong/panns_transfer_to_gtzan

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from augmentation import SpecAugmentModule
from features import FEATURE_DIM


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


# Attention

class ChannelAttention(nn.Module):
    """
    CBAM-style channel attention inserted after conv_block6.
    GlobalAvgPool then FC(2048→256)  then ReLU then FC(256→2048) then Sigmoid then scale input.
    """

    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, F)"""
        w = self.gap(x).view(x.size(0), -1)         # (B, C)
        w = self.fc(w).view(x.size(0), -1, 1, 1)    # (B, C, 1, 1)
        return x * w


class HandcraftedMLP(nn.Module):
    """
    Two-layer MLP that processes the 72-dim hand-crafted feature vector into
    a fixed 64-dim embedding that is fused with the CNN14 embedding.
    """

    def __init__(self, in_dim: int = FEATURE_DIM, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    """
    Fusion classification head.
    Input : concat(backbone embedding, MLP out [64]) -> (cnn_dim + 64)
    Output: log-softmax probabilities over classes_num classes.
    """

    def __init__(self, cnn_dim: int = 2048, mlp_dim: int = 64,
                 classes_num: int = 5, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cnn_dim + mlp_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes_num),
        )

    def forward(self, cnn_emb: torch.Tensor, mlp_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([cnn_emb, mlp_emb], dim=1)
        return torch.log_softmax(self.net(fused), dim=-1)


# ─── CNN14 building blocks ────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._init_weights()

    def _init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        return x


# CNN14 backbone

class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        # Spectrogram / log-mel extractor (frozen; part of the forward graph)
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8,  freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(1,    64)
        self.conv_block2 = ConvBlock(64,   128)
        self.conv_block3 = ConvBlock(128,  256)
        self.conv_block4 = ConvBlock(256,  512)
        self.conv_block5 = ConvBlock(512,  1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self._init_weights()

    def _init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """input: (batch, audio_length)"""
        x = self.spectrogram_extractor(input)   # (B, 1, T, F)
        x = self.logmel_extractor(x)            # (B, 1, T, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        if self.training and mixup_lambda is not None:
            from torch import Tensor
            x = (x[0::2].transpose(0, -1) * mixup_lambda[0::2]
                 + x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        if getattr(self, "channel_attention", None) is not None:
            x = self.channel_attention(x)

        x = torch.mean(x, dim=3)               # average over mel axis
        x1, _ = torch.max(x, dim=2)            # max-pool over time
        x2 = torch.mean(x, dim=2)              # avg-pool over time
        x = x1 + x2                            # (B, 2048)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class Transfer_Cnn14(nn.Module):

    # different freezing combinations to try in our second phase (best was partial 3 where we unfreeze last 3 layers)
    _UNFREEZE_MAP = {
        "all":      [],
        "partial1": ["conv_block6"],
        "partial2": ["conv_block5", "conv_block6"],
        "partial3": ["conv_block4", "conv_block5", "conv_block6"],
        "none":     ["conv_block1", "conv_block2", "conv_block3",
                     "conv_block4", "conv_block5", "conv_block6",
                     "fc1", "bn0"],
    }

    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: float,
        fmax: float,
        classes_num: int,
        freeze_layers: str = "all",
        use_attention: bool = False,
        use_feature_fusion: bool = False,
        use_specaugment: bool = False,
        specaugment_cfg: dict = None,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        audioset_classes_num = 527

        self.base = Cnn14(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, audioset_classes_num
        )

        # Spec Augment
        if use_specaugment and specaugment_cfg is not None:
            self.base.spec_augmenter = SpecAugmentModule(
                freq_mask_param=specaugment_cfg.get("freq_mask_param", 20),
                time_mask_param=specaugment_cfg.get("time_mask_param", 40),
                num_freq_masks=specaugment_cfg.get("num_freq_masks",   2),
                num_time_masks=specaugment_cfg.get("num_time_masks",   2),
            )
        else:
            # Disable the backbone's built-in SpecAugmentation entirely
            self.base.spec_augmenter = nn.Identity()

        # channel attention
        self.use_attention = use_attention
        if use_attention:
            self.base.channel_attention = ChannelAttention(in_channels=2048)

        # dual branch fusion head
        self.use_feature_fusion = use_feature_fusion
        if use_feature_fusion:
            self.handcrafted_mlp = HandcraftedMLP(in_dim=FEATURE_DIM,
                                                  dropout=fusion_dropout)
            self.fusion_head = FusionHead(cnn_dim=2048, mlp_dim=64,
                                              classes_num=classes_num,
                                              hidden_dim=fusion_hidden_dim,
                                              dropout=fusion_dropout)

        # fallback
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        init_layer(self.fc_transfer)

        # apply freezing
        self._freeze_layers_name = freeze_layers
        self._apply_freeze(freeze_layers)

    def _apply_freeze(self, freeze_layers: str):
        if freeze_layers not in self._UNFREEZE_MAP:
            raise ValueError(
                f"freeze_layers must be one of {list(self._UNFREEZE_MAP)}, "
                f"got '{freeze_layers}'"
            )

        # Start fully frozen
        for p in self.base.parameters():
            p.requires_grad = False

        # Unfreeze the specified backbone modules
        for name in self._UNFREEZE_MAP[freeze_layers]:
            module = getattr(self.base, name)
            for p in module.parameters():
                p.requires_grad = True

        self._freeze_layers_name = freeze_layers

    def set_freeze(self, freeze_layers: str):
        self._apply_freeze(freeze_layers)

    def get_parameter_groups(self, base_lr: float, backbone_lr_multiplier: float = 0.1) -> list:

        head_params = list(self.fc_transfer.parameters())
        if self.use_feature_fusion:
            head_params += list(self.handcrafted_mlp.parameters())
            head_params += list(self.fusion_head.parameters())
        if self.use_attention:
            head_params += list(self.base.channel_attention.parameters())

        backbone_params = [p for p in self.base.parameters() if p.requires_grad]

        groups = [{"params": head_params, "lr": base_lr}]
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": base_lr * backbone_lr_multiplier,
            })

        unfrozen_names = self._UNFREEZE_MAP.get(self._freeze_layers_name, [])
        print(
            f"  Parameter groups: head {sum(p.numel() for p in head_params):,} params @ lr={base_lr:.2e}"
            + (f" | backbone ({', '.join(unfrozen_names)}) "
               f"{sum(p.numel() for p in backbone_params):,} params "
               f"@ lr={base_lr * backbone_lr_multiplier:.2e}"
               if backbone_params else "")
        )
        return groups

    # checkpoint loading
    def load_from_pretrain(self, pretrained_checkpoint_path: str):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location="cpu",
                                weights_only=False)
        missing, unexpected = self.base.load_state_dict(
            checkpoint["model"], strict=False
        )
        phase3_prefixes = ("channel_attention.",)
        truly_missing = [k for k in missing
                         if not any(k.startswith(p) for p in phase3_prefixes)]
        if truly_missing:
            raise RuntimeError(
                f"Unexpected missing keys in pretrained checkpoint: {truly_missing[:5]}"
            )
        print(f"Loaded backbone weights from {pretrained_checkpoint_path}")

    # forward - this is where the fusion happens if using feature fusion
    # otherwise just pass through to base and apply transfer head to embedding

    def forward(self, input, handcrafted_features=None, mixup_lambda=None):
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict["embedding"]

        if self.use_feature_fusion and handcrafted_features is not None:
            mlp_out = self.handcrafted_mlp(handcrafted_features)
            clipwise_output = self.fusion_head(embedding, mlp_out)
        else:
            clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)

        output_dict["clipwise_output"] = clipwise_output
        return output_dict


def build_model(cfg: dict):
    """
    Instantiate the model according to the config.  The main decision point is the
    """
    aug_cfg = cfg.get("augmentation", {})

    fusion_hidden_dim = cfg.get("fusion_hidden_dim", 256)
    fusion_dropout = cfg.get("fusion_dropout", 0.3)

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
        use_specaugment=(
            cfg.get("use_augmentation", False)
            and aug_cfg.get("use_specaugment", False)
        ),
        specaugment_cfg=aug_cfg,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
    )
    model.load_from_pretrain(cfg["checkpoint_path"])
    return model
