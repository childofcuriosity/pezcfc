import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "analyze_results.py"
SPEC = importlib.util.spec_from_file_location("reported_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_checked_in_scores_reproduce_reported_result():
    rows = MODULE.load_scores(ROOT / "results" / "per_run_scores.csv")
    table = MODULE.aggregate(rows)
    best = max(table, key=lambda row: row["mean"])

    assert len(rows) == 312
    assert len({row["image_id"] for row in rows}) == 8
    assert {row["seed"] for row in rows} == {0, 1, 2}
    assert len(table) == 13
    assert best["threshold"] == pytest.approx(0.005)
    assert best["mean"] == pytest.approx(0.527952707062165)
    assert best["delta_vs_baseline"] == pytest.approx(0.018455856790145)


def test_experiment_snapshot_matches_recorded_digest():
    metadata = json.loads(
        (ROOT / "results" / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(
        (ROOT / "reference" / "optim_utils_experiment.py").read_bytes()
    ).hexdigest()

    assert digest == metadata["experiment_snapshot_sha256"]

