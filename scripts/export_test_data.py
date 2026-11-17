"""Export held-out SGSSC inputs from the original research archive, without models.

This maintainer tool requires the original archive via --source-root. It never
reads training/validation waveforms, changes splits, fits a scale, or runs a
model. Existing output artifacts are protected against accidental replacement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


SPLITS = ("test_seen", "test_unseen")
TEST_SEED = 20261117
MASK_COUNT = 358
LENGTH = 512
CLASSES = ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64",
           "GFSK", "CPFSK", "AM-DSB", "AM-SSB", "WBFM"]
INDEX = "data/generated/formal-v1/index.json"
PROTOCOL = "data/manifests/formal-v1/protocol.json"
NORMALIZATION = "data/generated/formal-v1/normalization.json"
BASE_MANIFEST = "data/manifests/formal-v1/base_manifest.csv"
COMBINED_MANIFEST = "data/manifests/sgssc-expanded-downstream-v1/combined_base_manifest.csv"
LABEL_MANIFEST = "data/manifests/sgssc-expanded-downstream-v1/run_0.json"
HISTORICAL_RESULTS = "outputs/m3/sgssc-clean-reference-test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stable_seed(*parts) -> int:
    """Same versioned encoding as the archived uwa_amc.data.noise.stable_seed."""
    payload = json.dumps(["uwa-amc/stable-seed/v1", parts], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def fixed_masks(base_ids: np.ndarray) -> np.ndarray:
    """Copy the validation-mask algorithm, with a new published test seed.

    The historical string namespace is retained for algorithm equivalence;
    these examples are test data. 0,0 are reference-only condition/view
    coordinates, not a physical SNR. True means the point is masked.
    """
    masks = torch.zeros((len(base_ids), LENGTH), dtype=torch.bool)
    for row, base_id in enumerate(base_ids):
        seed = stable_seed("sgssc-val-mask", TEST_SEED, str(base_id), 0, 0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        masks[row, torch.randperm(LENGTH, generator=generator)[:MASK_COUNT]] = True
    return masks.numpy()


def complex_spectrum(iq: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(iq)
    z = torch.complex(tensor[..., 0], tensor[..., 1])
    transformed = torch.fft.fftshift(torch.fft.fft(z, dim=1, norm="ortho"), dim=1)
    return torch.stack((transformed.real, transformed.imag), dim=-1).numpy()


def artifact(path: Path, output: Path, fields: dict) -> dict:
    return {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in fields.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original uwa-amc archive (read-only input).")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the read-only source archive")
    destinations = [output / "provenance/data_manifest.json"]
    for split in SPLITS:
        destinations.extend((output / f"data/classification/{split}.npz",
                             output / f"data/classification/{split}_metadata.csv",
                             output / f"data/spectrum/{split}.npz"))
    if any(path.exists() for path in destinations):
        raise FileExistsError("Export artifacts already exist; use a new --output-root")
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    index, protocol = read_json(source / INDEX), read_json(source / PROTOCOL)
    normalization = read_json(source / NORMALIZATION)
    if index["status"] != "complete" or protocol["labels"] != CLASSES:
        raise ValueError("Unexpected source protocol")
    if normalization["fit_split"] != "train" or index["scale"] != normalization["scale"]:
        raise ValueError("Source normalization is not the frozen train-fitted scale")
    if sha256(source / NORMALIZATION) != index["normalization_sha256"]:
        raise ValueError("Source normalization hash mismatch")

    with (source / BASE_MANIFEST).open(encoding="utf-8-sig", newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    test_rows = {split: {r["base_id"]: r for r in original_rows if r["split"] == split}
                 for split in SPLITS}
    blocked_ids, blocked_sources = set(), set()
    train_val_counts = {"train": 0, "val": 0}
    with (source / COMBINED_MANIFEST).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in train_val_counts:
                raise ValueError("Unexpected split in train/validation manifest")
            blocked_ids.add(row["base_id"])
            blocked_sources.add(row["source_id"])
            train_val_counts[row["split"]] += 1
    if train_val_counts != {"train": 242000, "val": 2200}:
        raise ValueError("Incomplete train/validation identity manifest")
    sets = [set(test_rows[split]) for split in SPLITS]
    if sets[0] & sets[1] or any(ids & blocked_ids for ids in sets):
        raise ValueError("Test identities overlap other splits")
    if any(row["source_id"] in blocked_sources for rows in test_rows.values() for row in rows.values()):
        raise ValueError("Test source identities overlap training/validation")
    if {r["source_id"] for r in test_rows[SPLITS[0]].values()} & {r["source_id"] for r in test_rows[SPLITS[1]].values()}:
        raise ValueError("Seen/unseen source identities overlap")

    source_files = [INDEX, PROTOCOL, NORMALIZATION, BASE_MANIFEST, COMBINED_MANIFEST,
                    LABEL_MANIFEST, "data/raw/pink/zenodo_record_21287414.json",
                    "src/uwa_amc/data/noise.py", "src/uwa_amc/models/sgssc.py",
                    "src/uwa_amc/training/sgssc.py",
                    "outputs/m3/sgssc-expanded-masked-run0/pretrain/report.json"]
    manifest = {
        "schema": "sgssc-underwater-amc-test-release-v1", "release_date": "2026-11-17",
        "scope": "held-out test data only; no training or validation waveform exports",
        "data_description": "Synthetic modulation waveforms replayed through measured Pink channel impulse responses; semi-synthetic received I/Q, not real recordings of the eleven modulations.",
        "input": {"sample_rate_hz": 8000, "window_length": LENGTH, "iq_order": ["I", "Q"],
                  "dtype": "float32", "source_field": "r", "noise_mode": "none_added",
                  "none_added_meaning": "No additional generated receiver noise; measured-channel propagation and channel-estimation effects remain."},
        "class_names": CLASSES,
        "normalization": {"scale": normalization["scale"], "source": NORMALIZATION,
                          "fit_split": "train", "fit_complex_samples": normalization["fit_complex_samples"],
                          "export_rescaling": "none; bitwise copy of stored float32 r"},
        "spectrum_test": {
            "status": "new fixed test task derived from previously held-out classification waveforms for this release",
            "historical_test_metrics_claimed": False,
            "historical_monitoring": "The pretraining report's 512 observations are validation monitoring, not this test set.",
            "seed": TEST_SEED, "mask_count": MASK_COUNT, "mask_ratio_requested": 0.7,
            "mask_ratio_actual": MASK_COUNT / LENGTH,
            "mask_semantics": "True=masked; both I and Q zeroed at identical positions",
            "mask_algorithm": "For each base ID: stable_seed('sgssc-val-mask',20261117,base_id,0,0), CPU torch.Generator.manual_seed, torch.randperm(512)[:358]. Independent of batch order and model outputs.",
            "seed_encoding": "JSON ['uwa-amc/stable-seed/v1',parts], ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False; first 8 SHA256 bytes big-endian, bitwise AND with 2**63-1.",
            "coordinates": "0,0 are reference-only condition/view indices, not physical SNR",
            "target": "All 512 complex bins of torch.fft.fftshift(torch.fft.fft(I+1j*Q,dim=1,norm='ortho'),dim=1), stacked real/imaginary float32; computed from complete unmasked r.",
            "labels_in_spectrum_archives": False,
        },
        "split_protocol": protocol["splits"],
        "independence_assumption": protocol["independence_assumption"],
        "temporal_boundary": protocol["temporal_boundary"],
        "isolation_audit": {"train_validation_manifest_counts": train_val_counts,
                            "test_base_id_overlap_with_train_val": 0,
                            "test_source_id_overlap_with_train_val": 0,
                            "seen_unseen_base_id_overlap": 0, "seen_unseen_source_id_overlap": 0,
                            "training_validation_waveforms_read": 0,
                            "model_inference_calls": 0},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "torch": torch.__version__, "h5py": h5py.__version__, "device": "cpu"},
        "splits": {},
    }
    record = read_json(source / "data/raw/pink/zenodo_record_21287414.json")
    metadata = record["metadata"]
    manifest["upstream_data"] = {
        "title": metadata["title"], "doi": metadata["doi"], "version": metadata["version"],
        "url": "https://zenodo.org/records/21287414", "license_id": metadata["license"]["id"],
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "creators": metadata["creators"],
        "license_evidence": "Saved official Zenodo record snapshot; original measured CIRs are transformed by signal replay, receiver preprocessing, crop extraction and fixed masking/FFT for this release.",
        "raw_channel_files_bundled": False,
    }

    for split in SPLITS:
        prediction_path = f"{HISTORICAL_RESULTS}/{split}-predictions.npz"
        source_files.append(prediction_path)
        # Read only identity and ground-truth fields. Model scores/predictions
        # play no role in ordering, selection, masking or any other export step.
        with np.load(source / prediction_path, allow_pickle=False) as historical:
            ids = historical["base_ids"].astype(str)
            historical_labels = historical["labels"].copy()
        if len(ids) != 2200 or len(set(ids)) != 2200 or set(ids) != set(test_rows[split]):
            raise ValueError("Historical test identity list is not the complete held-out split")
        position = {str(base_id): i for i, base_id in enumerate(ids)}
        iq = np.empty((len(ids), LENGTH, 2), dtype=np.float32)
        labels = np.empty(len(ids), dtype=np.int64)
        observed, locations, shard_entries = set(), {}, []
        for shard in index["shards"]:
            if shard["split"] != split:
                continue
            path = source / shard["path"]
            if sha256(path) != shard["sha256"]:
                raise ValueError(f"Source shard hash mismatch: {shard['path']}")
            with h5py.File(path, "r") as handle:
                shard_ids = handle["base_id"].asstr()[:]
                values, classes = handle["r"][:], handle["label_id"][:]
            if values.shape != (len(shard_ids), LENGTH, 2) or values.dtype != np.float32:
                raise ValueError("Unexpected source I/Q shape or dtype")
            for row_index, base_id in enumerate(shard_ids):
                if base_id in observed or base_id not in position:
                    raise ValueError("Duplicate or unexpected source test identity")
                observed.add(base_id)
                j = position[base_id]
                iq[j], labels[j] = values[row_index], classes[row_index]
                if int(test_rows[split][base_id]["modulation_id"]) != int(classes[row_index]):
                    raise ValueError("Manifest and source shard label disagreement")
                locations[base_id] = {"source_shard": shard["path"], "source_shard_row": row_index}
            shard_entries.append({"path": shard["path"], "sha256": shard["sha256"],
                                  "bytes": path.stat().st_size, "count": len(shard_ids), "field": "r"})
        if observed != set(ids) or not np.isfinite(iq).all():
            raise ValueError("Missing or nonfinite held-out inputs")
        np.testing.assert_array_equal(labels, historical_labels)
        np.testing.assert_array_equal(np.bincount(labels, minlength=11), np.full(11, 200))
        classification_path = output / f"data/classification/{split}.npz"
        classification = {"iq": iq, "labels": labels, "base_ids": ids}
        np.savez_compressed(classification_path, **classification)
        masks = fixed_masks(ids)
        masked = torch.from_numpy(iq).masked_fill(torch.from_numpy(masks).unsqueeze(-1), 0).numpy()
        target = complex_spectrum(iq)
        spectrum_path = output / f"data/spectrum/{split}.npz"
        spectrum = {"masked_iq": masked, "target_spectrum": target,
                    "masked_positions": masks, "base_ids": ids}
        np.savez_compressed(spectrum_path, **spectrum)

        # Verify serialization and every scalar against the in-memory export.
        for path, arrays in ((classification_path, classification), (spectrum_path, spectrum)):
            with np.load(path, allow_pickle=False) as stored:
                if set(stored.files) != set(arrays):
                    raise ValueError("Unexpected exported fields")
                for key, value in arrays.items():
                    np.testing.assert_array_equal(stored[key], value)
        # Second independent read of each historical HDF5 input checks copying.
        for shard in shard_entries:
            with h5py.File(source / shard["path"], "r") as handle:
                for row_index, base_id in enumerate(handle["base_id"].asstr()[:]):
                    np.testing.assert_array_equal(iq[position[base_id]], handle["r"][row_index])
        np.testing.assert_array_equal(masks.sum(axis=1), np.full(len(ids), MASK_COUNT))
        np.testing.assert_array_equal(masked[masks], np.zeros((len(ids) * MASK_COUNT, 2), np.float32))
        np.testing.assert_array_equal(masked[~masks], iq[~masks])
        np.testing.assert_array_equal(target, complex_spectrum(iq))
        np.testing.assert_array_equal(fixed_masks(ids[::-1])[::-1], masks)

        csv_path = output / f"data/classification/{split}_metadata.csv"
        fields = list(next(iter(test_rows[split].values()))) + ["source_shard", "source_shard_row"]
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for base_id in ids:
                writer.writerow({**test_rows[split][str(base_id)], **locations[str(base_id)]})
        manifest["splits"][split] = {
            "count": len(ids), "class_counts": {name: int((labels == i).sum()) for i, name in enumerate(CLASSES)},
            "order_source": prediction_path,
            "order_policy": "Complete historical held-out ID order; no subset selection and no model-score access",
            "source_shards": shard_entries,
            "classification": artifact(classification_path, output, classification),
            "spectrum": artifact(spectrum_path, output, spectrum),
            "sample_provenance": {"path": csv_path.relative_to(output).as_posix(),
                                  "bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)},
            "verification": {"source_iq_bitwise_equal": True, "historical_labels_equal": True,
                             "npz_roundtrip_arrays_equal": True, "mask_count_every_sample": MASK_COUNT,
                             "mask_batch_order_independent": True, "masked_iq_and_visible_values_exact": True,
                             "spectrum_target_recalculation_bitwise_equal": True},
        }
    manifest["source_files_sha256"] = {name: sha256(source / name) for name in source_files}
    manifest["exporter"] = {"path": "scripts/export_test_data.py", "sha256": sha256(Path(__file__))}
    destination = output / "provenance/data_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, info in manifest["splits"].items():
        print(json.dumps({"split": split, "count": info["count"],
                          "classification_bytes": info["classification"]["bytes"],
                          "spectrum_bytes": info["spectrum"]["bytes"]}))
    print("Saved provenance/data_manifest.json; no models or training/validation waveforms used.")


if __name__ == "__main__":
    main()
