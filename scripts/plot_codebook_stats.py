#!/usr/bin/env python3
"""Summarize and plot the saved per-token embedding standard deviations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/codebook_stds.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/codebook_stds.png"))
    args = parser.parse_args()

    values = np.load(args.input)
    print(f"tokens={len(values)}")
    print(f"min={values.min():.6f} max={values.max():.6f}")
    print(f"mean={values.mean():.6f} median={np.median(values):.6f}")
    for threshold in (0.0, 0.005, 0.01, 0.011, 0.012):
        kept = int((values >= threshold).sum())
        print(f"threshold={threshold:.3f}: kept={kept}/{len(values)} ({kept/len(values):.1%})")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(values, bins=180, color="#2457a7", edgecolor="none")
    ax.axvline(0.005, color="#c43c39", linestyle="--", label="reported best: 0.005")
    ax.set(xlabel="Embedding-coordinate std per token", ylabel="Token count")
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

