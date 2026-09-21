#!/usr/bin/env python3
"""Extract final scores from PEZ logs and anonymize image identifiers."""

import argparse
import csv
import re
from pathlib import Path


FILENAME = re.compile(
    r"^(?P<image>.+)_std_(?P<threshold>\d+(?:\.\d+)?)_seed_(?P<seed>\d+)\.log$"
)
FINAL_SCORE = re.compile(r"^best cosine sim:\s*([-+0-9.eE]+)\s*$", re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    parsed = []
    for path in sorted(args.log_dir.glob("*.log")):
        name_match = FILENAME.match(path.name)
        if not name_match:
            continue
        score_matches = FINAL_SCORE.findall(path.read_text(encoding="utf-8"))
        if not score_matches:
            raise ValueError(f"missing final score in {path}")
        parsed.append(
            (
                name_match.group("image"),
                int(name_match.group("seed")),
                float(name_match.group("threshold")),
                float(score_matches[-1]),
            )
        )

    if not parsed:
        raise ValueError(f"no matching logs found in {args.log_dir}")

    image_map = {
        name: f"image_{index:02d}"
        for index, name in enumerate(sorted({row[0] for row in parsed}), start=1)
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "seed", "threshold", "best_clip_similarity"))
        for image, seed, threshold, score in parsed:
            writer.writerow((image_map[image], seed, f"{threshold:.3f}", f"{score:.12f}"))

    print(
        f"wrote {len(parsed)} rows for {len(image_map)} anonymized images "
        f"to {args.output_csv}"
    )


if __name__ == "__main__":
    main()

