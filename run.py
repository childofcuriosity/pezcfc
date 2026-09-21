#!/usr/bin/env python3
"""Optimize a discrete prompt for one or more target images."""

import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PEZ with optional codebook-std candidate filtering."
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pez_vit_h14.json"),
        help="JSON experiment config (default: configs/pez_vit_h14.json)",
    )
    parser.add_argument(
        "--std-threshold",
        "--std_threshold",
        dest="std_threshold",
        type=float,
        default=0.0,
        help="Keep candidate tokens with embedding-coordinate std >= this value. "
        "Use 0 for the unfiltered PEZ baseline.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the iteration count in the JSON config (useful for smoke tests).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def main():
    cli = parse_args()

    # Delay heavyweight imports until image paths and CLI syntax are validated.
    images = [Image.open(path).convert("RGB") for path in cli.images]

    import torch
    import open_clip
    from optim_utils import optimize_prompt, read_json, set_random_seed

    args = argparse.Namespace(**read_json(str(cli.config)))
    args.std_threshold = cli.std_threshold
    args.print_new_best = True
    if cli.iterations is not None:
        if cli.iterations < 1:
            raise ValueError("--iterations must be positive")
        args.iter = cli.iterations

    set_random_seed(cli.seed)
    if cli.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cli.device

    print(f"seed: {cli.seed}")
    print(f"device: {device}")
    print(f"std threshold: {cli.std_threshold:.6f}")
    print(f"loading {args.clip_model} ({args.clip_pretrain})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrain, device=device
    )

    print(f"running for {args.iter} steps")
    learned_prompt = optimize_prompt(
        model, preprocess, args, device, target_images=images
    )
    print(learned_prompt)


if __name__ == "__main__":
    main()
