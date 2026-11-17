"""CPU training on externally supplied, no-added-noise train/validation data.

Adapted from the historical expanded SGSSC experiment, not the authors'
official implementation. Defaults retain its objectives, sampling streams,
masking and validation selection. No test data, noise generation or resume
path is provided. This portable NPY reader replaces project-specific HDF5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .inference import CLASS_NAMES
from .models import (SGSSCConfig, SGSSCEncoder, SGSSCReadout,
                     FlattenSpectralHead, mask_iq, spectrum_target)

LEGACY_CLEAN_INFERENCE_SHA256 = "31849ac8d35a78b3aa468b3de92444e3caf415cb1dcc4915d8be52301b687d70"


def stable_seed(*parts: object) -> int:
    """Preserve the original project's versioned seed derivation exactly."""
    encoded = json.dumps(["uwa-amc/stable-seed/v1", parts], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@dataclass(frozen=True)
class TrainingSettings:
    steps: int
    lr: float
    batch_size: int = 64
    validate_every: int = 250
    seed: int = 20261001
    threads: int = 4
    validation_observations: int = 512

    def __post_init__(self):
        for key in ("steps", "batch_size", "validate_every", "threads", "validation_observations"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative int64")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")


class CleanDataset:
    """Read only one declared train/val split from memory-mapped NPY arrays.

    manifest.json: schema='sgssc-clean-data-v1', split='train' or 'val',
    noise_mode='none_added', class_names=list(CLASS_NAMES). Files are iq.npy
    (float32 [N,L,2]), base_ids.npy (Unicode [N]), and labels.npy (int64 [N])
    only for classification. Pretraining never opens labels.npy. Array row
    order is preserved; a classifier train directory is the preselected
    labeled subset, not the entire unlabeled pool.
    """
    def __init__(self, directory: str | Path, split: str, length: int,
                 classification: bool = False):
        if split not in ("train", "val"):
            raise ValueError("Only train and val splits are accepted")
        self.directory = Path(directory).resolve()
        if self.directory.name.lower().startswith("test"):
            raise ValueError("Training readers do not accept test directories")
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("schema") != "sgssc-clean-data-v1"
                or self.manifest.get("split") != split
                or self.manifest.get("noise_mode") != "none_added"
                or tuple(self.manifest.get("class_names", ())) != CLASS_NAMES):
            raise ValueError("Expected the declared no-added-noise train/val manifest and class order")
        self.iq = np.load(self.directory / "iq.npy", mmap_mode="r", allow_pickle=False)
        self.base_ids = np.load(self.directory / "base_ids.npy", mmap_mode="r", allow_pickle=False)
        if (self.iq.dtype != np.float32 or self.iq.ndim != 3
                or self.iq.shape[1:] != (length, 2) or len(self.iq) < 1
                or self.base_ids.shape != (len(self.iq),) or self.base_ids.dtype.kind != "U"):
            raise ValueError("Expected float32 IQ [N,length,2] and Unicode base_ids [N]")
        identities = self.base_ids.tolist()
        if (len(set(identities)) != len(identities) or any(not x for x in identities)
                or any(x.lower().startswith("test") for x in identities)):
            raise ValueError("base_ids must be unique, nonempty training/validation identities")
        self.labels = None
        self.paths = [self.directory / name for name in ("manifest.json", "iq.npy", "base_ids.npy")]
        if classification:
            self.labels = np.load(self.directory / "labels.npy", mmap_mode="r", allow_pickle=False)
            if (self.labels.dtype != np.int64 or self.labels.shape != (len(self.iq),)
                    or np.any(self.labels < 0) or np.any(self.labels >= len(CLASS_NAMES))):
                raise ValueError("labels must be int64 [N] in the fixed 11-class order")
            self.paths.append(self.directory / "labels.npy")

    def __len__(self):
        return len(self.iq)

    def batch(self, rows: np.ndarray) -> torch.Tensor:
        values = np.array(self.iq[rows], dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite IQ data")
        return torch.from_numpy(values)

    def identity(self) -> dict:
        return {"directory": str(self.directory), "windows": len(self),
                "files_sha256": {path.name: _sha(path) for path in self.paths}}


def _check_disjoint(train: CleanDataset, val: CleanDataset) -> None:
    if set(train.base_ids.tolist()).intersection(val.base_ids.tolist()):
        raise ValueError("Training and validation base_ids overlap")


class CleanPretrainer(nn.Module):
    """70% masked IQ -> trainable encoder -> ordered linear spectrum head."""
    def __init__(self, config: SGSSCConfig):
        super().__init__()
        self.encoder = SGSSCEncoder(config)
        self.head = FlattenSpectralHead(config)

    def forward(self, masked_iq: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(masked_iq)[-1])


def validation_indices(count: int, observations: int, seed: int) -> np.ndarray:
    """Original no-added-noise validation selection, including its permutation."""
    if not 1 <= observations <= count:
        raise ValueError("validation_observations must not exceed the validation pool")
    rng = np.random.default_rng(stable_seed("sgssc-val-selection", seed))
    selected = rng.choice(count, observations, replace=False)
    return selected[rng.permutation(observations)]


def fixed_validation_mask(iq: torch.Tensor, base_ids, seed: int) -> torch.Tensor:
    """Use original per-base mask seeds; unaffected by batch order/size."""
    values = []
    for row, base_id in enumerate(base_ids):
        generator = torch.Generator(device="cpu").manual_seed(
            stable_seed("sgssc-val-mask", seed, str(base_id), 0, 0))
        values.append(mask_iq(iq[row:row + 1], 0.7, generator)[0])
    return torch.cat(values)


@torch.inference_mode()
def _spectrum_metrics(model: CleanPretrainer, masked: torch.Tensor,
                      target: torch.Tensor, batch_size: int) -> dict:
    model.eval()
    squared = energy = 0.0
    for start in range(0, len(masked), batch_size):
        prediction = model(masked[start:start + batch_size]).double()
        truth = target[start:start + batch_size].double()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite spectrum validation output")
        squared += float((prediction - truth).square().sum())
        energy += float(truth.square().sum())
    return {"mse": squared / target.numel(), "nmse": squared / energy if energy else None,
            "observations": len(masked)}


def _setup(settings: TrainingSettings) -> None:
    torch.set_num_threads(settings.threads)
    random.seed(settings.seed)
    np.random.seed(settings.seed % 2**32)
    torch.random.default_generator.manual_seed(settings.seed)


def _event(stream, event: dict) -> None:
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()


def _rng_state(sampler, generator=None) -> dict:
    # Tensor and primitive-only payload: torch.load(weights_only=True) remains sufficient.
    return {"torch": torch.get_rng_state(), "sampler": sampler.bit_generator.state,
            "mask_generator": None if generator is None else generator.get_state()}


def train_pretrain(train_data: str | Path, val_data: str | Path, output: str | Path,
                   config: SGSSCConfig | None = None,
                   settings: TrainingSettings | None = None) -> dict:
    config = config or SGSSCConfig()
    settings = settings or TrainingSettings(steps=20000, lr=1e-4)
    train = CleanDataset(train_data, "train", config.length)
    val = CleanDataset(val_data, "val", config.length)
    _check_disjoint(train, val)
    selected = validation_indices(len(val), settings.validation_observations, settings.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _setup(settings)
    model = CleanPretrainer(config)
    initial_encoder = _state_hash(model.encoder)
    initial_head = _state_hash(model.head)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=0)
    sampler = np.random.default_rng(stable_seed("sgssc-pretrain-sampling", settings.seed))
    mask_generator = torch.Generator(device="cpu").manual_seed(stable_seed("sgssc-training-mask", settings.seed))
    original = val.batch(selected)
    masked = fixed_validation_mask(original, val.base_ids[selected], settings.seed)
    target = spectrum_target(original)
    run = {"stage": "pretrain", "noise_mode": "none_added", "settings": asdict(settings),
           "model_config": asdict(config), "train": train.identity(), "validation": val.identity(),
           "validation_base_ids": val.base_ids[selected].tolist(), "training_labels_used": 0,
           "validation_labels_used": 0, "test_observations_used": 0,
           "selection_rule": "minimum validation MSE; earliest tie; includes step 0",
           "mask_ratio": 0.7, "sampling": "uniform with replacement per update",
           "optimizer": "Adam; weight_decay=0", "torch": str(torch.__version__), "numpy": np.__version__}
    _write_json(output / "run.json", run)
    best, best_step, best_metrics = math.inf, -1, None
    counts = np.zeros(len(train), dtype=np.int64)
    started = time.perf_counter()
    with (output / "curve.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(settings.steps + 1):
            if step:
                model.train()
                rows = sampler.integers(len(train), size=settings.batch_size)
                iq = train.batch(rows)
                inputs, _ = mask_iq(iq, 0.7, mask_generator)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(inputs), spectrum_target(iq))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite reconstruction loss")
                loss.backward()
                optimizer.step()
                np.add.at(counts, rows, 1)
                _event(stream, {"event": "train", "step": step, "loss": float(loss.detach())})
            if step % settings.validate_every == 0 or step == settings.steps:
                metrics = _spectrum_metrics(model, masked, target, settings.batch_size)
                _event(stream, {"event": "validation", "step": step, **metrics})
                payload = {"schema": "sgssc-clean-pretrain-v1", "model_config": asdict(config),
                           "encoder_state": model.encoder.state_dict(), "spectral_head_state": model.head.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "rng": _rng_state(sampler, mask_generator),
                           "step": step, "seed": settings.seed, "validation": metrics,
                           "metadata": {"noise_mode": "none_added", "training": run}}
                if metrics["mse"] < best:
                    best, best_step, best_metrics = metrics["mse"], step, metrics
                    _save(output / "best.pt", payload)
                if step == settings.steps:
                    _save(output / "last.pt", payload)
    np.savez_compressed(output / "training_exposures.npz", base_ids=train.base_ids, counts=counts)
    report = {**run, "steps": settings.steps, "selected_step": best_step,
              "best_validation": best_metrics, "last_validation": metrics,
              "encoder_updated": initial_encoder != _state_hash(model.encoder),
              "spectral_head_updated": initial_head != _state_hash(model.head),
              "optimization_observations": int(counts.sum()), "seconds": time.perf_counter() - started,
              "best_sha256": _sha(output / "best.pt"), "last_sha256": _sha(output / "last.pt")}
    _write_json(output / "report.json", report)
    return report


def load_pretrained(path: str | Path):
    """Load only encoder/spectrum states, never a previous classifier head."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema") not in ("sgssc-clean-pretrain-v1", "sgssc-inference-v1"):
        raise ValueError("Expected a portable clean pretraining or released inference checkpoint")
    if state["schema"] == "sgssc-inference-v1" and tuple(state.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("Checkpoint class order mismatch")
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("A declared no-added-noise training source is required")
    declared_clean = metadata.get("noise_mode") == "none_added"
    # The first public release predates this declaration. Its fixed, audited
    # bytes are the only missing-field exception; unknown sources are rejected.
    known_legacy = (state["schema"] == "sgssc-inference-v1" and "noise_mode" not in metadata
                    and _sha(Path(path)) == LEGACY_CLEAN_INFERENCE_SHA256)
    if not (declared_clean or known_legacy):
        raise ValueError("Only explicitly no-added-noise or hash-verified legacy clean sources are accepted")
    config = SGSSCConfig(**state["model_config"])
    with torch.random.fork_rng(devices=[]):
        encoder, head = SGSSCEncoder(config), FlattenSpectralHead(config)
    for module, key in ((encoder, "encoder_state"), (head, "spectral_head_state")):
        tensors = state[key]
        if not isinstance(tensors, dict) or any(not isinstance(x, torch.Tensor) or not torch.isfinite(x).all() for x in tensors.values()):
            raise ValueError("Checkpoint must contain finite tensor states")
        module.load_state_dict(tensors, strict=True)
    encoder.requires_grad_(False).eval()
    head.requires_grad_(False).eval()
    return config, encoder, head


@torch.inference_mode()
def _cache_features(encoder: SGSSCEncoder, dataset: CleanDataset, path: Path, batch_size: int):
    config = encoder.config
    encoder.eval()
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                     shape=(len(dataset), config.depth, config.tokens, config.dim))
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        values[start:stop] = encoder.encode_all(dataset.batch(np.arange(start, stop))).numpy()
    values.flush()
    del values
    return np.load(path, mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def _classifier_metrics(readout, features, labels, batch_size):
    readout.eval()
    confusion = np.zeros((11, 11), dtype=np.int64)
    losses, predictions, all_logits = 0.0, [], []
    for start in range(0, len(labels), batch_size):
        logits = readout(torch.from_numpy(np.array(features[start:start + batch_size], copy=True)))
        target = torch.from_numpy(np.array(labels[start:start + batch_size], copy=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite classification validation output")
        losses += float(F.cross_entropy(logits, target, reduction="sum"))
        predicted = logits.argmax(1).numpy()
        np.add.at(confusion, (target.numpy(), predicted), 1)
        predictions.append(predicted)
        all_logits.append(logits.numpy())
    denominators = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * np.diag(confusion), denominators, out=np.zeros(11), where=denominators != 0)
    correct = int(np.trace(confusion))
    return {"loss": losses / len(labels), "correct": correct, "accuracy": correct / len(labels),
            "macro_f1": float(f1.mean()), "confusion": confusion.tolist(), "observations": len(labels)}, {
                "labels": np.asarray(labels), "predictions": np.concatenate(predictions), "logits": np.concatenate(all_logits)}


def train_classifier(train_data: str | Path, val_data: str | Path, pretrained: str | Path,
                     output: str | Path, settings: TrainingSettings | None = None,
                     label_fraction: str = "0.005", training_pool_size: int = 242000) -> dict:
    settings = settings or TrainingSettings(steps=2000, lr=1e-3)
    config, encoder, spectral_head = load_pretrained(pretrained)
    train = CleanDataset(train_data, "train", config.length, classification=True)
    val = CleanDataset(val_data, "val", config.length, classification=True)
    _check_disjoint(train, val)
    fraction = float(label_fraction)
    if (not math.isfinite(fraction) or not 0 < fraction <= 1 or type(training_pool_size) is not int
            or training_pool_size < 1 or len(train) != round(training_pool_size * fraction)):
        raise ValueError("Preselected training subset size must equal round(training_pool_size * label_fraction)")
    # Canonical decimal text preserves the original fraction-specific RNG stream.
    label_fraction = format(fraction, ".12g")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _setup(settings)
    encoder.requires_grad_(False).eval()
    frozen_hash = _state_hash(encoder)
    train_features = _cache_features(encoder, train, output / "train_features.npy", settings.batch_size)
    val_features = _cache_features(encoder, val, output / "val_features.npy", settings.batch_size)
    torch.random.default_generator.manual_seed(stable_seed("sgssc-readout-initialization", settings.seed))
    readout = SGSSCReadout(config)
    initial_readout = _state_hash(readout)
    optimizer = torch.optim.Adam(readout.parameters(), lr=settings.lr, weight_decay=0)
    sampler = np.random.default_rng(stable_seed("sgssc-readout-sampling", settings.seed, label_fraction))
    run = {"stage": "classifier", "noise_mode": "none_added", "settings": asdict(settings),
           "model_config": asdict(config), "train": train.identity(), "validation": val.identity(),
           "training_labels": len(train), "validation_labels_separately": len(val),
           "label_fraction": label_fraction, "training_pool_size": training_pool_size,
           "pretrained_sha256": _sha(Path(pretrained)), "encoder_frozen": True,
           "selection_rule": "maximum full-validation correct count; earliest tie; includes step 0",
           "classification_input": "complete IQ; no masking; no PCA", "objective": "cross_entropy",
           "optimizer": "Adam; weight_decay=0", "sampling": "uniform with replacement per update",
           "test_observations_used": 0, "torch": str(torch.__version__), "numpy": np.__version__}
    _write_json(output / "run.json", run)
    best_correct, best_step, best_metrics = -1, -1, None
    counts = np.zeros(len(train), dtype=np.int64)
    started = time.perf_counter()
    with (output / "curve.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(settings.steps + 1):
            if step:
                readout.train()
                rows = sampler.integers(len(train), size=settings.batch_size)
                features = torch.from_numpy(np.array(train_features[rows], copy=True))
                labels = torch.from_numpy(np.array(train.labels[rows], copy=True))
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(readout(features), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite cross-entropy loss")
                loss.backward()
                optimizer.step()
                np.add.at(counts, rows, 1)
                _event(stream, {"event": "train", "step": step, "loss": float(loss.detach())})
            if step % settings.validate_every == 0 or step == settings.steps:
                metrics, predictions = _classifier_metrics(readout, val_features, val.labels, settings.batch_size)
                _event(stream, {"event": "validation", "step": step, **metrics})
                if (encoder.training or any(p.requires_grad or p.grad is not None for p in encoder.parameters())
                        or _state_hash(encoder) != frozen_hash):
                    raise RuntimeError("Encoder freeze invariant failed")
                payload = {"schema": "sgssc-inference-v1", "model_config": asdict(config),
                           "class_names": list(CLASS_NAMES), "encoder_state": encoder.state_dict(),
                           "readout_state": readout.state_dict(), "spectral_head_state": spectral_head.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "rng": _rng_state(sampler), "step": step,
                           "metadata": {"noise_mode": "none_added", "training": run, "validation": metrics}}
                if metrics["correct"] > best_correct:
                    best_correct, best_step, best_metrics = metrics["correct"], step, metrics
                    _save(output / "best.pt", payload)
                    np.savez_compressed(output / "best_validation_predictions.npz", **predictions, base_ids=val.base_ids)
                if step == settings.steps:
                    _save(output / "last.pt", payload)
    np.savez_compressed(output / "training_exposures.npz", base_ids=train.base_ids, counts=counts)
    report = {**run, "steps": settings.steps, "selected_step": best_step, "best_validation": best_metrics,
              "last_validation": metrics, "encoder_unchanged": _state_hash(encoder) == frozen_hash,
              "readout_updated": initial_readout != _state_hash(readout),
              "optimization_observations": int(counts.sum()), "seconds": time.perf_counter() - started,
              "best_sha256": _sha(output / "best.pt"), "last_sha256": _sha(output / "last.pt")}
    _write_json(output / "report.json", report)
    return report
