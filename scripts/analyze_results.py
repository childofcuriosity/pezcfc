#!/usr/bin/env python3
"""Aggregate the threshold sweep and optionally render its key figure."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


def load_scores(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "image_id": row["image_id"],
                    "seed": int(row["seed"]),
                    "threshold": float(row["threshold"]),
                    "score": float(row["best_clip_similarity"]),
                }
            )
    if not rows:
        raise ValueError(f"no scores in {path}")
    return rows


def aggregate(rows):
    by_threshold_seed = defaultdict(list)
    for row in rows:
        by_threshold_seed[(row["threshold"], row["seed"])].append(row["score"])

    seed_means = {
        key: fmean(values) for key, values in by_threshold_seed.items()
    }
    thresholds = sorted({key[0] for key in seed_means})
    baseline = fmean(
        value for (threshold, _), value in seed_means.items() if threshold == 0.0
    )

    table = []
    for threshold in thresholds:
        values = [
            value for (candidate, _), value in seed_means.items()
            if candidate == threshold
        ]
        run_count = sum(row["threshold"] == threshold for row in rows)
        table.append(
            {
                "threshold": threshold,
                "n_runs": run_count,
                "mean": fmean(values),
                "std_across_seed_means": stdev(values) if len(values) > 1 else math.nan,
                "delta_vs_baseline": fmean(values) - baseline,
            }
        )
    return table


def write_table(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "threshold",
            "n_runs",
            "mean",
            "std_across_seed_means",
            "delta_vs_baseline",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow(
                {
                    "threshold": f"{row['threshold']:.3f}",
                    "n_runs": row["n_runs"],
                    "mean": f"{row['mean']:.6f}",
                    "std_across_seed_means": f"{row['std_across_seed_means']:.6f}",
                    "delta_vs_baseline": f"{row['delta_vs_baseline']:+.6f}",
                }
            )


def plot_table(table, path):
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in table]
    means = [row["mean"] for row in table]
    errors = [row["std_across_seed_means"] for row in table]
    best = max(range(len(means)), key=means.__getitem__)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(
        thresholds,
        means,
        yerr=errors,
        marker="o",
        capsize=3,
        linewidth=1.7,
        color="#2457a7",
    )
    ax.scatter([thresholds[best]], [means[best]], color="#c43c39", zorder=3)
    ax.annotate(
        f"best: {thresholds[best]:.3f}\n{means[best]:.4f}",
        (thresholds[best], means[best]),
        xytext=(12, 8),
        textcoords="offset points",
    )
    ax.axhline(means[0], color="#666666", linestyle="--", linewidth=1, label="baseline")
    ax.set(xlabel="Codebook std threshold", ylabel="Best CLIP cosine similarity")
    ax.set_ylim(top=max(means) + max(errors) + 0.012)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("results/per_run_scores.csv"))
    parser.add_argument("--table", type=Path, default=Path("results/threshold_sweep.csv"))
    parser.add_argument("--figure", type=Path, default=Path("results/threshold_sweep.png"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    table = aggregate(load_scores(args.scores))
    write_table(table, args.table)
    if not args.no_plot:
        plot_table(table, args.figure)

    best = max(table, key=lambda row: row["mean"])
    print(
        f"best threshold={best['threshold']:.3f}, mean={best['mean']:.6f}, "
        f"delta={best['delta_vs_baseline']:+.6f}, n={best['n_runs']}"
    )


if __name__ == "__main__":
    main()
