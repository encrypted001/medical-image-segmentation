"""A beginner-friendly, configurable U-Net for 2D image segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """U-Net encoder/decoder with feature skip connections.

    Args:
        in_channels: Number of input channels (1 for grayscale MRI slices).
        out_channels: Number of output channels (1 for binary segmentation).
        base_channels: First encoder width; later levels double this value.
        feature_sizes: Optional explicit width for each encoder level, e.g.
            ``(16, 32, 64, 128)``. Overrides ``base_channels`` when provided.

    The forward pass returns raw logits. Apply sigmoid for binary probabilities
    or softmax across channels for multiclass probabilities outside the model.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        feature_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")

        features = tuple(feature_sizes) if feature_sizes is not None else tuple(
            base_channels * (2**level) for level in range(4)
        )
        if len(features) < 2 or any(width < 1 for width in features):
            raise ValueError("feature_sizes must contain at least two positive channel widths")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_sizes = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: each level learns features, then max pooling halves the
        # spatial size while the next level increases the channel width.
        encoder_blocks = []
        previous_channels = in_channels
        for width in features:
            encoder_blocks.append(DoubleConv(previous_channels, width))
            previous_channels = width
        self.encoders = nn.ModuleList(encoder_blocks)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: transposed convolutions upsample. Each upsampled feature map
        # is concatenated with its matching encoder feature (the skip
        # connection), restoring spatial detail lost through pooling.
        upsamplers = []
        decoder_blocks = []
        decoder_channels = features[-1] * 2
        for width in reversed(features):
            upsamplers.append(nn.ConvTranspose2d(decoder_channels, width, 2, stride=2))
            decoder_blocks.append(DoubleConv(width * 2, width))
            decoder_channels = width
        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoders = nn.ModuleList(decoder_blocks)
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shaped [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Model expects {self.in_channels} input channel(s), got {x.shape[1]}"
            )
        minimum_size = 2 ** len(self.feature_sizes)
        if min(x.shape[-2:]) < minimum_size:
            raise ValueError(
                f"Input height and width must be at least {minimum_size} for "
                f"{len(self.feature_sizes)} encoder levels"
            )

        original_size = x.shape[-2:]
        skips = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if level < len(self.encoders) - 1:
                x = self.pool(x)

        x = self.bottleneck(self.pool(x))
        for upsample, decoder, skip in zip(
            self.upsamplers, self.decoders, reversed(skips)
        ):
            x = upsample(x)
            # Odd input dimensions can cause pooling to round down. Resize the
            # upsampled map to its skip's exact size before concatenation.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))

        logits = self.output(x)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits
