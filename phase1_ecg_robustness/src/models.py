"""Compact clean-baseline architectures; all outputs are unnormalized logits."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ResNet1D(nn.Module):
    """ResNet18-style 2,2,2,2 blocks, narrower than an image ResNet."""

    def __init__(
        self, base_channels: int = 24, in_channels: int = 12, num_classes: int = 5
    ):
        super().__init__()
        if min(base_channels, in_channels, num_classes) < 1:
            raise ValueError("Channel and class counts must be positive")
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks = []
        previous = base_channels
        for stage, width in enumerate(
            (base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels)
        ):
            blocks.append(ResidualBlock(previous, width, 1 if stage == 0 else 2))
            blocks.append(ResidualBlock(width, width))
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(previous, num_classes)
        self.apply(_initialize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.blocks(self.stem(x))).squeeze(-1))


class CausalConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, 3, dilation=dilation, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, dropout: float
    ):
        super().__init__()
        # GroupNorm is independent of batch composition (normalization pools time).
        self.layers = nn.Sequential(
            CausalConv1D(in_channels, out_channels, dilation),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1D(out_channels, out_channels, dilation),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layers(x) + self.shortcut(x))


class TCN(nn.Module):
    """Six dilated residual blocks; convolution-only receptive field 253 samples.

    Convolutions are left-padded; GroupNorm uses the complete ECG window, so this
    classifier is not advertised as an online/strictly causal predictor.
    """

    def __init__(
        self,
        base_channels: int = 24,
        in_channels: int = 12,
        num_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(base_channels, in_channels, num_classes) < 1:
            raise ValueError("Channel and class counts must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        blocks = []
        previous = in_channels
        for i, multiplier in enumerate((1, 1, 2, 2, 4, 4)):
            width = base_channels * multiplier
            blocks.append(TemporalBlock(previous, width, 2**i, dropout))
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(previous, num_classes)
        self.apply(_initialize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.blocks(x)).squeeze(-1))


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(name: str, **kwargs) -> nn.Module:
    """Construct a baseline from the exact name and kwargs saved in a checkpoint."""
    constructors = {"resnet": ResNet1D, "tcn": TCN}
    if name not in constructors:
        raise ValueError(f"Unknown model {name!r}; choose one of {list(constructors)}")
    return constructors[name](**kwargs)
