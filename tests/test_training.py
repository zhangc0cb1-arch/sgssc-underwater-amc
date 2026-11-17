"""Small synthetic checks only: no project training/validation/test waveforms."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sgssc_amc.inference import CLASS_NAMES, load_checkpoint
from sgssc_amc.models import SGSSCConfig, SGSSCReadout, mask_iq, spectrum_target
from sgssc_amc.training import (CleanDataset, CleanPretrainer, TrainingSettings,
                                fixed_validation_mask, load_pretrained, stable_seed,
                                train_classifier, train_pretrain, validation_indices)


SMALL = SGSSCConfig(length=32, patch=8, dim=16, depth=2, heads=4, ff=32, dropout=0.1)


def dataset(path: Path, split: str, count: int):
    path.mkdir()
    manifest = {"schema": "sgssc-clean-data-v1", "split": split,
                "noise_mode": "none_added", "class_names": list(CLASS_NAMES)}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    values = np.random.default_rng(7 if split == "train" else 8).normal(size=(count, 32, 2)).astype(np.float32)
    np.save(path / "iq.npy", values)
    np.save(path / "base_ids.npy", np.asarray([f"{split}-{i}" for i in range(count)]))
    np.save(path / "labels.npy", np.arange(count, dtype=np.int64) % 11)
    return path


@pytest.fixture()
def data_dirs(tmp_path):
    return dataset(tmp_path / "train", "train", 6), dataset(tmp_path / "val", "val", 4)


def test_pretext_gradient_reaches_encoder_and_head():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = CleanPretrainer(SMALL)
    iq = torch.randn(3, 32, 2)
    inputs, mask = mask_iq(iq, 0.7, torch.Generator().manual_seed(4))
    assert mask.sum(1).tolist() == [22] * 3
    torch.nn.functional.mse_loss(model(inputs), spectrum_target(iq)).backward()
    for module in (model.encoder, model.head):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_validation_masks_follow_ids_not_batch_order():
    values = torch.arange(4 * 32 * 2, dtype=torch.float32).reshape(4, 32, 2) + 1
    ids = ["val-a", "val-b", "val-c", "val-d"]
    expected = fixed_validation_mask(values, ids, 20261001)
    order = [2, 0, 3, 1]
    actual = fixed_validation_mask(values[order], [ids[i] for i in order], 20261001)
    torch.testing.assert_close(actual, expected[order], rtol=0, atol=0)
    assert (expected == 0).all(2).sum(1).tolist() == [22] * 4
    selected = validation_indices(20, 7, 20261001)
    rng = np.random.default_rng(stable_seed("sgssc-val-selection", 20261001))
    old = rng.choice(20, 7, replace=False)
    np.testing.assert_array_equal(selected, old[rng.permutation(7)])


def test_pretrain_never_opens_class_labels(data_dirs, monkeypatch):
    real_load = np.load
    def guarded(path, *args, **kwargs):
        assert Path(path).name != "labels.npy"
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded)
    data = CleanDataset(data_dirs[0], "train", 32)
    assert data.labels is None
    assert data.batch(np.array([0, 2])).shape == (2, 32, 2)


def test_reader_rejects_test_and_added_noise(data_dirs):
    with pytest.raises(ValueError, match="Only train and val"):
        CleanDataset(data_dirs[0], "test_seen", 32)
    manifest_path = data_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["noise_mode"] = "pink"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no-added-noise"):
        CleanDataset(data_dirs[0], "train", 32)


def test_split_overlap_rejected_before_output(tmp_path, data_dirs):
    np.save(data_dirs[1] / "base_ids.npy", np.asarray(["train-0", "val-1", "val-2", "val-3"]))
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="overlap"):
        train_pretrain(*data_dirs, output, SMALL,
                       TrainingSettings(steps=1, lr=1e-4, validation_observations=4))
    assert not output.exists()


def source_checkpoint(path, schema, metadata):
    from dataclasses import asdict
    model = CleanPretrainer(SMALL)
    torch.save({"schema": schema, "model_config": asdict(SMALL),
                "class_names": list(CLASS_NAMES), "encoder_state": model.encoder.state_dict(),
                "spectral_head_state": model.head.state_dict(),
                "readout_state": SGSSCReadout(SMALL).state_dict(), "metadata": metadata}, path)


@pytest.mark.parametrize("schema,metadata", [
    ("sgssc-clean-pretrain-v1", {}),
    ("sgssc-inference-v1", {}),
    ("sgssc-clean-pretrain-v1", {"noise_mode": "pink"}),
    ("sgssc-inference-v1", {"noise_mode": None}),
])
def test_pretrained_source_requires_clean_declaration(tmp_path, schema, metadata):
    path = tmp_path / "unknown.pt"
    source_checkpoint(path, schema, metadata)
    with pytest.raises(ValueError, match="clean sources"):
        load_pretrained(path)


def test_missing_noise_field_legacy_exception_is_hash_limited(tmp_path, monkeypatch):
    import sgssc_amc.training as training
    path = tmp_path / "legacy.pt"
    source_checkpoint(path, "sgssc-inference-v1", {})
    monkeypatch.setattr(training, "_sha", lambda _: training.LEGACY_CLEAN_INFERENCE_SHA256)
    _, encoder, _ = load_pretrained(path)
    assert not encoder.training
    # Even the legacy hash exception must not override an explicit noisy source.
    source_checkpoint(path, "sgssc-inference-v1", {"noise_mode": "pink"})
    with pytest.raises(ValueError, match="clean sources"):
        load_pretrained(path)


def test_two_stage_updates_freeze_selection_and_safe_reload(tmp_path, data_dirs):
    pre_dir, classifier_dir = tmp_path / "pretrain", tmp_path / "classifier"
    options = TrainingSettings(steps=2, lr=1e-3, batch_size=3, validate_every=1,
                               threads=1, validation_observations=4)
    pre_report = train_pretrain(*data_dirs, pre_dir, SMALL, options)
    assert pre_report["encoder_updated"] and pre_report["spectral_head_updated"]
    assert pre_report["training_labels_used"] == pre_report["test_observations_used"] == 0
    assert pre_report["optimization_observations"] == 6
    pre_events = [json.loads(x) for x in (pre_dir / "curve.jsonl").read_text().splitlines()]
    validations = [x for x in pre_events if x["event"] == "validation"]
    assert [x["step"] for x in validations] == [0, 1, 2]
    assert pre_report["selected_step"] == min(validations, key=lambda x: x["mse"])["step"]
    source = torch.load(pre_dir / "best.pt", weights_only=True, map_location="cpu")
    _, encoder, head = load_pretrained(pre_dir / "best.pt")
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    assert all(torch.equal(v, source["encoder_state"][k]) for k, v in encoder.state_dict().items())
    report = train_classifier(*data_dirs, pre_dir / "best.pt", classifier_dir,
                              options, label_fraction="0.5", training_pool_size=12)
    assert report["encoder_unchanged"] and report["readout_updated"]
    assert report["training_labels"] == 6 and report["validation_labels_separately"] == 4
    assert report["optimization_observations"] == 6
    events = [json.loads(x) for x in (classifier_dir / "curve.jsonl").read_text().splitlines()]
    validations = [x for x in events if x["event"] == "validation"]
    assert report["selected_step"] == max(validations, key=lambda x: x["correct"])["step"]
    for filename in ("best.pt", "last.pt"):
        state = torch.load(classifier_dir / filename, weights_only=True, map_location="cpu")
        assert all(torch.equal(v, source["encoder_state"][k]) for k, v in state["encoder_state"].items())
        assert all(torch.equal(v, source["spectral_head_state"][k]) for k, v in state["spectral_head_state"].items())
        model, _ = load_checkpoint(classifier_dir / filename)
        iq = torch.from_numpy(np.array(np.load(data_dirs[1] / "iq.npy")[:3], copy=True))
        expected_features = torch.from_numpy(np.array(np.load(classifier_dir / "val_features.npy")[:3], copy=True))
        with torch.inference_mode():
            torch.testing.assert_close(model.classify(iq), model.readout(expected_features), rtol=0, atol=0)
        assert not model.training and all(not p.requires_grad for p in model.parameters())
    with np.load(classifier_dir / "best_validation_predictions.npz", allow_pickle=False) as saved:
        best, _ = load_checkpoint(classifier_dir / "best.pt")
        iq = torch.from_numpy(np.array(np.load(data_dirs[1] / "iq.npy"), copy=True))
        actual = torch.cat([best.classify(iq[i:i + 3]) for i in range(0, 4, 3)]).numpy()
        np.testing.assert_array_equal(actual, saved["logits"])
    with pytest.raises(FileExistsError):
        train_pretrain(*data_dirs, pre_dir, SMALL, options)
    with pytest.raises(ValueError, match="subset size"):
        train_classifier(*data_dirs, pre_dir / "best.pt", tmp_path / "wrong_budget", options)
    assert not (tmp_path / "wrong_budget").exists()
