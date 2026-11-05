"""Run a dummy batch through U-Net and print the resulting logits shape."""

from __future__ import annotations

import argparse

import torch

from src.model import UNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()

    model = UNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels,
    )
    dummy = torch.randn(args.batch_size, args.in_channels, args.height, args.width)
    with torch.no_grad():
        logits = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print("Output values are raw logits (no sigmoid applied).")


if __name__ == "__main__":
    main()
