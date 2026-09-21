import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_results", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_uses_std_across_seed_level_image_means():
    rows = [
        {"image_id": image, "seed": seed, "threshold": threshold, "score": score}
        for image, seed, threshold, score in [
            ("a", 0, 0.0, 0.4),
            ("b", 0, 0.0, 0.6),
            ("a", 1, 0.0, 0.5),
            ("b", 1, 0.0, 0.7),
            ("a", 0, 0.1, 0.6),
            ("b", 0, 0.1, 0.8),
            ("a", 1, 0.1, 0.7),
            ("b", 1, 0.1, 0.9),
        ]
    ]

    table = MODULE.aggregate(rows)

    assert table[0]["mean"] == 0.55
    assert table[1]["mean"] == 0.75
    assert table[1]["delta_vs_baseline"] == pytest.approx(0.20)
