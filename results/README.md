# Result artifacts

This directory contains the privacy-safe evidence used by the project README.

- `per_run_scores.csv`: all 312 final scores, with the eight source images
  replaced by stable anonymous IDs. It contains no image content or optimized
  prompt text.
- `threshold_sweep.csv`: the table regenerated from `per_run_scores.csv`.
- `threshold_sweep.png`: the corresponding figure. Error bars are the sample
  standard deviation across three seed-level means; each seed-level mean is
  computed over eight images.
- `codebook_stds.npy`: the 49,408 per-token embedding-coordinate standard
  deviations from the evaluated CLIP codebook.
- `codebook_stds.png`: the distribution generated from the NumPy file.
- `experiment_metadata.json`: model, optimization, environment, aggregation,
  and evaluation-scope details.

Regenerate the checked-in table and figures:

```bash
python scripts/analyze_results.py
python scripts/plot_codebook_stats.py
```

The source images and raw logs are intentionally excluded because the image
set includes private or redistribution-uncleared material, and raw logs contain
optimized prompt text. Consequently, the checked-in scores and analysis are
exactly reproducible, while the original 312 GPU runs are not independently
reproducible from this public repository alone.

