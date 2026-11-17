"""Read the published held-out NPZ arrays without pickle or training data."""
from __future__ import annotations

from pathlib import Path
import numpy as np

SPLITS = ("test_seen", "test_unseen")


def _coordinates(values: np.ndarray, count: int, length: int, name: str) -> None:
    if values.dtype != np.float32 or values.shape != (count, length, 2):
        raise ValueError(f"{name} must be float32 [{count}, {length}, 2]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")


def load_test_data(data_dir: str | Path, task: str, split: str,
                   length: int = 512) -> tuple[dict[str, np.ndarray], Path]:
    if task not in ("classification", "spectrum") or split not in SPLITS:
        raise ValueError("Only published classification/spectrum held-out splits are supported")
    path = Path(data_dir) / task / f"{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    required = ({"iq", "labels", "base_ids"} if task == "classification" else
                {"masked_iq", "target_spectrum", "masked_positions", "base_ids"})
    if not required.issubset(values):
        raise ValueError(f"{path.name}: missing fields {sorted(required - values.keys())}")
    ids = values["base_ids"]
    if ids.ndim != 1 or ids.dtype.kind != "U" or not len(ids):
        raise ValueError("base_ids must be a nonempty one-dimensional Unicode array")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate base_ids in held-out data")
    count = len(ids)
    if task == "classification":
        _coordinates(values["iq"], count, length, "iq")
        labels = values["labels"]
        if labels.dtype != np.int64 or labels.shape != (count,):
            raise ValueError("labels must be int64 [N]")
        if np.any((labels < 0) | (labels >= 11)):
            raise ValueError("Labels must be in [0, 10]")
    else:
        _coordinates(values["masked_iq"], count, length, "masked_iq")
        _coordinates(values["target_spectrum"], count, length, "target_spectrum")
        mask = values["masked_positions"]
        if mask.dtype != np.bool_ or mask.shape != (count, length):
            raise ValueError("masked_positions must be bool [N, L]")
        if not np.all(mask.sum(axis=1) == round(0.7 * length)):
            raise ValueError("Every spectral example must have the fixed 70% point-mask count")
        if np.any(values["masked_iq"][mask] != 0):
            raise ValueError("Masked I/Q coordinates must be exactly zero")
    return values, path

