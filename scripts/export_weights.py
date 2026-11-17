"""Export the two pinned research checkpoints as a safe inference-only bundle.

This optional release-maintenance script reads trusted, hash-verified source
checkpoints. It is not needed to run classification or spectrum evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct


SOURCES = {
    "pretraining": {
        "path": "outputs/m3/sgssc-expanded-masked-run0/pretrain/best.pt",
        "sha256": "85fa038cae33df7e9fc354fd0e2d7e8501867bcc2bec537bb5405c160b073216",
    },
    "classification": {
        "path": "outputs/m3/sgssc-expanded-downstream-run0/downstream-0.005/best.pt",
        "sha256": "fec272213352f56c7beaa5c38b053eca2d837bad287b64c2d41fb0007246cec1",
    },
}
CLASS_NAMES = ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64", "GFSK",
               "CPFSK", "AM-DSB", "AM-SSB", "WBFM"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensors(state: dict) -> dict:
    if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                            for k, v in state.items()):
        raise ValueError("Expected a nonempty tensor state dictionary")
    return {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    loaded = {}
    for role, source in SOURCES.items():
        path = args.source_root / source["path"]
        actual = sha256(path)
        if actual != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {role}")
        # Legacy training snapshots contain a NumPy MT19937 RNG array. Only
        # allow its specific constructors after checking the pinned file hash.
        # Never use weights_only=False. The exported bundle needs no allowlist.
        with torch.serialization.safe_globals([
            _reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType,
        ]):
            loaded[role] = torch.load(path, map_location="cpu", weights_only=True)
    pretrain, classifier = loaded["pretraining"], loaded["classification"]
    if pretrain["step"] != 20000 or classifier["step"] != 1000:
        raise ValueError("Unexpected selected training steps")
    if pretrain["model_config"] != classifier["model_config"]:
        raise ValueError("Model configurations do not match")
    if classifier["pretrain_checkpoint_sha256"] != SOURCES["pretraining"]["sha256"]:
        raise ValueError("Classifier provenance does not reference this pretraining checkpoint")
    pre_encoder = {k.removeprefix("encoder."): v for k, v in pretrain["model_state"].items()
                   if k.startswith("encoder.")}
    encoder = tensors(classifier["encoder_state"])
    if pre_encoder.keys() != encoder.keys() or any(
        not torch.equal(pre_encoder[k], encoder[k]) for k in encoder
    ):
        raise ValueError("The supposedly frozen encoders differ")
    head = tensors({k.removeprefix("head."): v for k, v in pretrain["model_state"].items()
                    if k.startswith("head.")})
    readout = tensors(classifier["readout_state"])
    if set(head) != {"projection.weight", "projection.bias"}:
        raise ValueError("Expected the ordered Flatten-to-Linear spectral head")
    metadata = {
        "implementation": "Independent SGSSC adaptation for underwater acoustic signals",
        "input_condition": "Measured-channel replay, no added noise, preserved training-fitted scale",
        "input_shape": [512, 2], "sample_rate_hz": 8000,
        "normalization_scale": 3.0439862190625893,
        "input_scale_note": "Bundled I/Q arrays are already scaled; do not apply the scale again.",
        "class_names": CLASS_NAMES, "pretraining_windows": 242000,
        "pretraining_updates": 20000, "pretraining_selected_step": 20000,
        "classification_training_labels": 1210, "validation_labels_separately": 2200,
        "classification_updates": 2000, "classification_selected_step": 1000,
        "seed": 20261001, "label_run": 0, "encoder_frozen_for_classification": True,
        "spectral_head_type": "flatten_linear", "pretraining_mask_ratio": 0.7,
        "sources": SOURCES,
    }
    bundle = {
        "schema": "sgssc-inference-v1", "model_config": dict(classifier["model_config"]),
        "class_names": CLASS_NAMES, "encoder_state": encoder,
        "readout_state": readout, "spectral_head_state": head, "metadata": metadata,
    }
    output = args.destination / "weights/sgssc_inference.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release bundle: {output}")
    torch.save(bundle, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    for name in ("encoder_state", "readout_state", "spectral_head_state"):
        if bundle[name].keys() != reloaded[name].keys() or any(
            not torch.equal(t, reloaded[name][k]) for k, t in bundle[name].items()
        ):
            raise ValueError("Export changed a source model tensor")
    counts = {name: sum(v.numel() for v in bundle[name].values())
              for name in ("encoder_state", "readout_state", "spectral_head_state")}
    manifest = {
        "schema": "sgssc-inference-weights-manifest-v1", "created_date": "2026-11-17",
        "file": "weights/sgssc_inference.pt", "bytes": output.stat().st_size,
        "sha256": sha256(output), "tensor_counts": counts, "metadata": metadata,
        "source_encoder_tensors_identical": True, "exported_tensors_identical": True,
        "safe_load_without_allowlist": True, "optimizer_or_rng_exported": False,
        "source_loading": "weights_only=True with explicit NumPy RNG allowlist after SHA256 verification",
        "export_environment": {"torch": torch.__version__, "numpy": np.__version__},
    }
    path = args.destination / "provenance/weights_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(output), "bytes": output.stat().st_size,
                      "tensor_counts": counts, "safe_load": True}, indent=2))


if __name__ == "__main__":
    main()
