from dataclasses import asdict

import numpy as np
import pytest
import torch

from sgssc_amc.cli import main
from sgssc_amc.data import load_test_data
from sgssc_amc.inference import CLASS_NAMES, InferenceModel, load_checkpoint
from sgssc_amc.metrics import classification_metrics, shifted_spectrum, spectrum_metrics
from sgssc_amc.models import SGSSCConfig, mask_iq, spectrum_target


@pytest.fixture
def public_checkpoint(tmp_path):
    config = SGSSCConfig(length=32, patch=16, dim=16, depth=2, heads=4, ff=32, dropout=0.1)
    torch.manual_seed(41)
    model = InferenceModel(config)
    path = tmp_path / "model.pt"
    torch.save({"schema": "sgssc-inference-v1", "model_config": asdict(config),
                "class_names": list(CLASS_NAMES), "encoder_state": model.encoder.state_dict(),
                "readout_state": model.readout.state_dict(),
                "spectral_head_state": model.spectral_head.state_dict(), "metadata": {"fixture": True}}, path)
    return path


def test_fixed_point_mask_and_complex_fft_convention():
    t = torch.arange(32, dtype=torch.float32)
    value = torch.stack((torch.cos(2 * torch.pi * 3 * t / 32),
                         torch.sin(2 * torch.pi * 3 * t / 32)), dim=-1)[None]
    masked, mask = mask_iq(value, generator=torch.Generator().manual_seed(43))
    assert mask.dtype == torch.bool
    assert mask.sum().item() == round(0.7 * 32)
    assert torch.equal(masked[~mask], value[~mask])
    assert torch.count_nonzero(masked[mask]).item() == 0
    spectrum = spectrum_target(value)
    assert spectrum.shape == value.shape
    np.testing.assert_allclose(spectrum.numpy(), shifted_spectrum(value.numpy()), rtol=1e-5, atol=1e-6)
    assert spectrum.square().sum().item() == pytest.approx(value.square().sum().item(), rel=1e-6)
    assert spectrum.square().sum(-1).argmax().item() == 16 + 3


def test_tensor_only_checkpoint_reloads_deterministic_frozen_heads(public_checkpoint):
    torch.set_num_threads(1)
    before = torch.random.get_rng_state().clone()
    model, metadata = load_checkpoint(public_checkpoint)
    assert torch.equal(before, torch.random.get_rng_state())
    again, _ = load_checkpoint(public_checkpoint)
    iq = torch.randn(3, 32, 2, generator=torch.Generator().manual_seed(2))
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    model.train()  # Inference-only mode cannot accidentally enable dropout.
    assert not any(module.training for module in model.modules())
    classification, spectrum = model.classify(iq), model.recover_spectrum(iq)
    assert classification.shape == (3, 11)
    assert spectrum.shape == (3, 32, 2)
    assert torch.equal(classification, again.classify(iq))
    assert torch.equal(spectrum, again.recover_spectrum(iq))
    assert not classification.requires_grad and not spectrum.requires_grad
    assert metadata["class_names"] == list(CLASS_NAMES)


def test_metrics_include_absent_classes_and_defined_spectral_baselines():
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), ["A", "B", "C"])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"] == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    iq = np.random.default_rng(2).normal(size=(2, 32, 2)).astype(np.float32)
    target = shifted_spectrum(iq)
    errors = spectrum_metrics(target, target, iq)
    assert errors["results"]["model"]["mse"] == 0
    assert errors["results"]["model"]["non_dc"]["nmse"] == 0
    assert errors["results"]["zero"]["nmse"] == 1
    assert errors["results"]["masked_input_fft"]["nmse"] == 0
    zeros = np.zeros_like(iq)
    assert spectrum_metrics(zeros, zeros, zeros)["results"]["model"]["nmse"] is None


def test_cli_uses_only_public_test_arrays(public_checkpoint, tmp_path):
    import json
    data_dir = tmp_path / "data"
    values = np.random.default_rng(8).normal(size=(2, 32, 2)).astype(np.float32)
    mask_values, mask = mask_iq(torch.from_numpy(values), generator=torch.Generator().manual_seed(9))
    for task in ("classification", "spectrum"):
        (data_dir / task).mkdir(parents=True)
    ids = np.array(["test_seen-example-0", "test_seen-example-1"])
    np.savez_compressed(data_dir / "classification/test_seen.npz", iq=values,
                        labels=np.array([0, 1], dtype=np.int64), base_ids=ids)
    np.savez_compressed(data_dir / "spectrum/test_seen.npz", masked_iq=mask_values.numpy(),
                        masked_positions=mask.numpy(), target_spectrum=spectrum_target(torch.from_numpy(values)).numpy(),
                        base_ids=ids)
    for command, task in (("classify", "classification"), ("spectrum", "spectrum")):
        output = tmp_path / "results" / task
        assert main([command, "--split", "test_seen", "--data-dir", str(data_dir),
                     "--checkpoint", str(public_checkpoint), "--output", str(output),
                     "--batch-size", "1", "--threads", "1", "--no-plots"]) == 0
        report = json.loads((output / "report.json").read_text())
        assert report["task"] == task
        assert report["training_updates"] == 0
        assert report["results"]["test_seen"]["metrics"]["observations"] == 2
        assert (output / "test_seen/predictions.npz").is_file()
    # The public loader rejects incorrectly encoded masks before inference.
    broken = mask_values.numpy().copy()
    broken[mask.numpy()] = 1
    np.savez_compressed(data_dir / "spectrum/test_seen.npz", masked_iq=broken,
                        masked_positions=mask.numpy(), target_spectrum=spectrum_target(torch.from_numpy(values)).numpy(),
                        base_ids=ids)
    with pytest.raises(ValueError, match="exactly zero"):
        load_test_data(data_dir, "spectrum", "test_seen", length=32)

