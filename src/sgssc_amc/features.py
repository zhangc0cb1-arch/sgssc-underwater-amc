"""No-added-noise SGSSC classifier features: row L2 -> PCA50 -> 2D t-SNE.

Every window in one held-out split participates in its own descriptive fit.
Labels only support plotting and evaluation; they never enter dimensionality
reduction. This visualization does not change the classification pathway.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import SPLITS, load_test_data
from .inference import CLASS_NAMES, load_checkpoint
from .metrics import classification_metrics


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _seed(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be in [0, 2**32 - 1]")
    return parsed


def add_visualize_parser(commands, root: Path) -> None:
    command = commands.add_parser("visualize", help="Plot no-added-noise classifier features with PCA50 and t-SNE")
    command.add_argument("--split", choices=SPLITS, default="test_unseen",
                         help="One held-out split per fit (default: test_unseen; all windows retained)")
    command.add_argument("--data-dir", type=Path, default=root / "data")
    command.add_argument("--checkpoint", type=Path, default=root / "weights/sgssc_inference.pt")
    command.add_argument("--output", type=Path, default=Path("results/features"))
    command.add_argument("--batch-size", type=_positive, default=64)
    command.add_argument("--threads", type=_positive, default=4)
    command.add_argument("--seed", type=_seed, default=20261031)
    command.add_argument("--max-iter", type=_positive, default=1500)
    command.add_argument("--no-plots", action="store_true", help="Write arrays, CSV and report without figures")


def classifier_tsne(features: np.ndarray, seed: int = 20261031,
                    threads: int = 4, max_iter: int = 1500):
    """Fit the original classifier-feature visualization protocol to one split.

    Row normalization uses float64 and epsilon 1e-12, followed by centered,
    unwhitened full-SVD PCA50. t-SNE uses its own PCA initialization. All rows
    retain source order. No labels, sampling, fitting across splits, or model
    training are involved. Returns ``coordinates, fitted_pca, metadata``.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    values = np.asarray(features, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 128 or len(values) < 50
            or not np.isfinite(values).all()):
        raise ValueError("Expected at least 50 finite classifier feature vectors with 128 dimensions")
    if not 0 <= seed <= 2**32 - 1 or threads < 1 or max_iter < 250:
        raise ValueError("seed must be in [0, 2**32 - 1], threads positive, and max_iter at least 250")
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    if np.sum(np.var(normalized, axis=0)) <= np.finfo(np.float64).eps:
        raise ValueError("PCA requires nonconstant L2-normalized classifier features")
    pca = PCA(n_components=50, svd_solver="full", whiten=False)
    options = dict(n_components=2, perplexity=30.0, early_exaggeration=12.0,
                   learning_rate="auto", max_iter=max_iter, init="pca", metric="euclidean",
                   method="barnes_hut", angle=0.5, random_state=seed, n_jobs=threads)
    estimator = TSNE(**options)
    with threadpool_limits(limits=threads):
        reduced = pca.fit_transform(normalized)
        coordinates = estimator.fit_transform(reduced)
    if (coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all()
            or not np.isfinite(estimator.kl_divergence_)):
        raise ValueError("t-SNE produced invalid coordinates or KL divergence")
    metadata = {
        "algorithm": "row L2 -> PCA50 -> 2D t-SNE", "feature_dimensions": 128,
        "normalization": {"kind": "row L2", "epsilon": 1e-12, "dtype": "float64"},
        "pca": {"components": 50, "svd_solver": "full", "whiten": False,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())},
        "tsne": {**options, "kl_divergence": float(estimator.kl_divergence_),
                 "n_iter": int(estimator.n_iter_)},
        "windows": len(values), "threads": threads,
        "fit_scope": "all windows in this held-out split, fitted independently of any other split",
        "axis_meaning": "embedding coordinates; no fixed physical units",
    }
    return coordinates, pca, metadata


def _write_coordinates(path, coordinates, labels, split, ids, predictions):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature_row", "base_id", "split", "label", "class_name",
                         "prediction", "predicted_class", "tSNE1", "tSNE2"])
        for row, point in enumerate(coordinates):
            writer.writerow([row, ids[row], split, int(labels[row]), CLASS_NAMES[labels[row]],
                             int(predictions[row]), CLASS_NAMES[predictions[row]], *map(float, point)])


def run_visualization(args: argparse.Namespace) -> dict:
    try:
        import sklearn
    except ImportError as error:
        raise SystemExit('Install visualization dependencies: python -m pip install -e ".[visualize]"') from error
    from .cli import sha256, write_json

    if args.split not in SPLITS:
        raise ValueError("Visualize one published no-added-noise held-out split at a time")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    model, checkpoint_info = load_checkpoint(args.checkpoint)
    arrays, source_path = load_test_data(args.data_dir, "classification", args.split, model.config.length)
    directory = args.output / args.split
    directory.mkdir(parents=True, exist_ok=True)
    feature_batches, logit_batches = [], []
    print(f"Extracting classifier features: {args.split}, all {len(arrays['labels'])} windows", flush=True)
    for start in range(0, len(arrays["labels"]), args.batch_size):
        result = model.extract_features(torch.from_numpy(arrays["iq"][start:start + args.batch_size]))
        feature_batches.append(result["classifier_features"].cpu().numpy())
        logit_batches.append(result["logits"].cpu().numpy())
    features, logits = np.concatenate(feature_batches), np.concatenate(logit_batches)
    if not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise ValueError("Nonfinite classifier features or logits")
    predictions = logits.argmax(axis=1).astype(np.int64)
    print("Fitting L2 -> PCA50 -> t-SNE on this split (no sample selection)", flush=True)
    coordinates, pca, embedding = classifier_tsne(features, args.seed, args.threads, args.max_iter)
    np.savez_compressed(directory / "features.npz", classifier_features=features,
                        logits=logits, predictions=predictions, labels=arrays["labels"],
                        base_ids=arrays["base_ids"], coordinates=coordinates)
    np.savez_compressed(directory / "pca.npz", mean=pca.mean_, components=pca.components_,
                        explained_variance=pca.explained_variance_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    _write_coordinates(directory / "coordinates.csv", coordinates, arrays["labels"],
                       args.split, arrays["base_ids"], predictions)
    artifacts = {"features": "features.npz", "pca": "pca.npz", "coordinates": "coordinates.csv"}
    if not args.no_plots:
        from .feature_plots import write_feature_plots
        artifacts.update(write_feature_plots(coordinates, arrays["labels"], directory))
    report = {
        "schema": "sgssc-classifier-tsne-v1", "training_updates": 0, "added_noise": False,
        "split": args.split, "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_schema": checkpoint_info["schema"], "model_config": asdict(model.config),
        "data": {"file": source_path.name, "sha256": sha256(source_path), "windows": len(predictions)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": str(torch.__version__), "scikit_learn": sklearn.__version__},
        "device": "cpu", "threads": args.threads, "batch_size": args.batch_size,
        "class_names": list(CLASS_NAMES),
        "class_counts": {name: int(np.sum(arrays["labels"] == index)) for index, name in enumerate(CLASS_NAMES)},
        "sampling": "all supplied held-out windows in source order; no sampling or correctness filtering",
        "label_use": "color and classification evaluation only; no labels in normalization, PCA, or t-SNE",
        "interpretation": "Exploratory transductive visualization, not classification accuracy or a generalization score",
        "classification_path": "complete IQ -> frozen encoder -> attention readout (128D) -> linear classifier; no PCA or t-SNE",
        "classification_all_windows": classification_metrics(arrays["labels"], predictions, CLASS_NAMES),
        "embedding": embedding,
        "artifacts": {name: {"file": f"{args.split}/{filename}", "sha256": sha256(directory / filename)}
                      for name, filename in artifacts.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(f"Report: {(args.output / 'report.json').resolve()}", flush=True)
    return report
