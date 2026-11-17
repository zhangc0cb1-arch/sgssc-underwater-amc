"""Run frozen CPU evaluation or exploratory feature visualization."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import SPLITS, load_test_data
from .inference import CLASS_NAMES, load_checkpoint
from .metrics import classification_metrics, spectrum_metrics

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="task", required=True)
    for task, description in (("classify", "Classify complete held-out I/Q windows"),
                              ("spectrum", "Recover full spectra from fixed masked held-out I/Q")):
        command = commands.add_parser(task, help=description)
        command.add_argument("--split", choices=(*SPLITS, "all"), default="all")
        command.add_argument("--data-dir", type=Path, default=ROOT / "data",
                             help="Parent of classification/ and spectrum/ (default: repository data/)")
        command.add_argument("--checkpoint", type=Path, default=ROOT / "weights/sgssc_inference.pt")
        command.add_argument("--output", type=Path, default=Path("results") / task)
        command.add_argument("--batch-size", type=positive_integer, default=64)
        command.add_argument("--threads", type=positive_integer, default=4)
        command.add_argument("--no-plots", action="store_true", help="Write numeric results only")
    from .features import add_visualize_parser
    add_visualize_parser(commands, ROOT)
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    model, checkpoint_info = load_checkpoint(args.checkpoint)
    checkpoint_hash = sha256(args.checkpoint)
    split_names = SPLITS if args.split == "all" else (args.split,)
    task_data = "classification" if args.task == "classify" else "spectrum"
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "sgssc-held-out-evaluation-v1", "task": task_data,
              "device": "cpu", "threads": args.threads, "batch_size": args.batch_size,
              "checkpoint_sha256": checkpoint_hash, "checkpoint_schema": checkpoint_info["schema"],
              "model_config": asdict(model.config), "class_names": list(CLASS_NAMES),
              "checkpoint_metadata": checkpoint_info["metadata"],
              "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
              "training_updates": 0, "selection_policy": "fixed pretrained weights; no test-driven selection",
              "results": {}}
    all_ids: set[str] = set()
    for split in split_names:
        arrays, source_path = load_test_data(args.data_dir, task_data, split, model.config.length)
        ids = arrays["base_ids"]
        if all_ids.intersection(ids.tolist()):
            raise ValueError("Held-out Seen/Unseen split IDs overlap")
        all_ids.update(ids.tolist())
        directory = args.output / split
        directory.mkdir(parents=True, exist_ok=True)
        input_values = arrays["iq" if args.task == "classify" else "masked_iq"]
        outputs = []
        for start in range(0, len(ids), args.batch_size):
            batch = torch.from_numpy(input_values[start:start + args.batch_size])
            output = model.classify(batch) if args.task == "classify" else model.recover_spectrum(batch)
            outputs.append(output.cpu().numpy())
        predicted = np.concatenate(outputs).astype(np.float32, copy=False)
        if not np.isfinite(predicted).all():
            raise ValueError("Model returned nonfinite outputs")
        if args.task == "classify":
            labels = predicted.argmax(axis=1).astype(np.int64)
            metrics = classification_metrics(arrays["labels"], labels, CLASS_NAMES)
            np.savez_compressed(directory / "predictions.npz", base_ids=ids, logits=predicted,
                                predictions=labels, labels=arrays["labels"])
            if not args.no_plots:
                from .plots import confusion_plot
                confusion_plot(metrics, directory / "confusion_matrix.png", f"SGSSC: {split} (no added noise)")
            print(f"{split}: accuracy={metrics['accuracy']:.4%}, macro-F1={metrics['macro_f1']:.4%}", flush=True)
        else:
            metrics = spectrum_metrics(predicted, arrays["target_spectrum"], input_values)
            metrics["mask"] = {"requested_ratio": 0.7, "points_per_window": round(0.7 * model.config.length),
                               "actual_ratio": round(0.7 * model.config.length) / model.config.length,
                               "policy": "fixed saved point masks; shared I/Q mask; True means masked"}
            np.savez_compressed(directory / "predictions.npz", base_ids=ids, predicted_spectrum=predicted)
            if not args.no_plots:
                from .plots import spectrum_plot
                spectrum_plot(predicted[0], arrays["target_spectrum"][0], input_values[0],
                              str(ids[0]), directory / "spectrum_example.png")
            result = metrics["results"]["model"]
            print(f"{split}: MSE={result['mse']:.6g}, NMSE={result['nmse']}, "
                  f"non-DC NMSE={result['non_dc']['nmse']}", flush=True)
        write_json(directory / "metrics.json", metrics)
        report["results"][split] = {"data_file": source_path.name, "data_sha256": sha256(source_path),
                                    "metrics": metrics,
                                    "artifacts": {"metrics": f"{split}/metrics.json",
                                                  "predictions": f"{split}/predictions.npz"}}
        if not args.no_plots:
            plot_name = "confusion_matrix.png" if args.task == "classify" else "spectrum_example.png"
            report["results"][split]["artifacts"]["plot"] = f"{split}/{plot_name}"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["timing_scope"] = "checkpoint/data reading, inference, metrics and optional plotting; not a latency benchmark"
    write_json(args.output / "report.json", report)
    print(f"Report: {(args.output / 'report.json').resolve()}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.task == "visualize":
        from .features import run_visualization
        run_visualization(arguments)
    else:
        run(arguments)
    return 0

