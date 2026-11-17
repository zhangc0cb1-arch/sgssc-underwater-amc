"""Redraw the clean SGSSC reference without extracting features or fitting t-SNE."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from sgssc_amc.feature_plots import write_feature_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinates', type=Path, default=(
        ROOT / 'reference_results/features/test_unseen/coordinates.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/reference_features'))
    args = parser.parse_args()
    with args.coordinates.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('coordinate table is empty')
    labels = np.array([int(row['label']) for row in rows], dtype=np.int64)
    coordinates = np.array([[float(row['tSNE1']), float(row['tSNE2'])]
                            for row in rows], dtype=np.float64)
    files = write_feature_plots(coordinates, labels, args.output)
    for filename in files.values():
        print((args.output / filename).resolve())


if __name__ == '__main__':
    main()
