"""Feature extraction preserves the public inference/checkpoint contract.

Only synthetic tensors and a temporary tensor-only checkpoint are used. These
tests neither require nor read the original research archive or released data.
"""
from dataclasses import asdict
import argparse
import csv
from types import SimpleNamespace
import inspect

import numpy as np
import pytest
import torch

from sgssc_amc.inference import CLASS_NAMES, InferenceModel, load_checkpoint
from sgssc_amc.models import SGSSCConfig


LEGACY_READOUT_KEYS = {
    "attention.0.weight", "attention.0.bias", "attention.2.weight",
    "attention.2.bias", "classifier.weight", "classifier.bias",
}


@pytest.fixture
def feature_model(tmp_path):
    """Use the unchanged original checkpoint schema, without any feature state."""
    config = SGSSCConfig(length=64, patch=16, dim=16, depth=2,
                         heads=4, ff=32, dropout=0.1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261117)
        original = InferenceModel(config)
    state = original.readout.state_dict()
    assert set(state) == LEGACY_READOUT_KEYS
    checkpoint = tmp_path / "legacy_inference.pt"
    torch.save({"schema": "sgssc-inference-v1", "model_config": asdict(config),
                "class_names": list(CLASS_NAMES),
                "encoder_state": original.encoder.state_dict(),
                "readout_state": state,
                "spectral_head_state": original.spectral_head.state_dict(),
                "metadata": {"fixture": "original inference schema without feature fields"}},
               checkpoint)
    loaded, _ = load_checkpoint(checkpoint)
    return loaded


@pytest.fixture
def feature_inputs():
    generator = torch.Generator(device="cpu").manual_seed(117)
    return torch.randn(5, 64, 2, generator=generator)


def test_extracted_levels_and_logits_preserve_classifier(feature_model, feature_inputs):
    model, iq = feature_model, feature_inputs
    features = model.extract_features(iq)
    assert set(features) == {"encoder_tokens", "classifier_features", "logits"}
    assert features["encoder_tokens"].shape == (5, 4, 16)
    assert features["classifier_features"].shape == (5, 16)
    assert features["logits"].shape == (5, 11)
    assert torch.equal(features["logits"], model.classify(iq))

    # Independently retain the pre-refactor pooling definition: all encoder
    # layers contribute and token gates are not divided by their weight sum.
    with torch.inference_mode():
        layers = model.encoder(iq)
        fused = torch.cat(layers, dim=1)
        weights = model.readout.attention(fused.mean(dim=-1))
        original_pooled = (fused * weights.unsqueeze(-1)).sum(dim=1)
        original_logits = model.readout.classifier(original_pooled)
    assert torch.equal(features["encoder_tokens"], layers[-1])
    assert torch.equal(features["classifier_features"], original_pooled)
    assert torch.equal(features["logits"], original_logits)


def test_features_keep_sample_order_across_batch_sizes(feature_model, feature_inputs):
    model, iq = feature_model, feature_inputs
    whole = model.extract_features(iq)
    parts = [model.extract_features(iq[:2]), model.extract_features(iq[2:4]),
             model.extract_features(iq[4:])]
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = model.extract_features(iq[order])
    inverse = torch.argsort(order)
    for name, expected in whole.items():
        # CPU linear algebra can round differently for different batch sizes;
        # numerical equality rather than byte equality is the contract here.
        torch.testing.assert_close(torch.cat([p[name] for p in parts]), expected,
                                   rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(permuted[name][inverse], expected,
                                   rtol=1e-5, atol=2e-6)


def test_readout_accepts_both_existing_layer_layouts(feature_model, feature_inputs):
    model = feature_model
    with torch.inference_mode():
        layers = model.encoder(feature_inputs)
        pooled_list = model.readout.extract_features(layers)
        pooled_stack = model.readout.extract_features(torch.stack(layers, dim=1))
        original_logits = model.readout(layers)
    assert torch.equal(pooled_list, pooled_stack)
    assert torch.equal(model.readout.classifier(pooled_list), original_logits)


def test_feature_api_adds_no_checkpoint_parameters_or_state_updates(feature_model, feature_inputs):
    model = feature_model
    assert set(model.readout.state_dict()) == LEGACY_READOUT_KEYS
    before = {key: value.clone() for key, value in model.state_dict().items()}
    original_spectrum = model.recover_spectrum(feature_inputs)
    model.extract_features(feature_inputs)
    after = model.state_dict()
    assert set(after) == set(before)
    assert all(torch.equal(value, before[key]) for key, value in after.items())
    assert torch.equal(model.recover_spectrum(feature_inputs), original_spectrum)
    # Strict loading of the same pre-existing tensor schema remains possible.
    fresh = InferenceModel(model.config)
    fresh.load_state_dict(before, strict=True)
    assert torch.equal(fresh.classify(feature_inputs), model.classify(feature_inputs))


def test_feature_extraction_stays_inference_only_and_reads_no_files(
        feature_model, feature_inputs, monkeypatch):
    model = feature_model
    model.train()
    assert all(not module.training for module in model.modules())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    model.extract_features(feature_inputs)  # Initialize CPU kernels before the I/O guard.
    iq = feature_inputs.clone().requires_grad_(True)

    def reject_file_access(*args, **kwargs):
        raise AssertionError("Tensor feature extraction must not read a data/checkpoint file")

    with monkeypatch.context() as guard:
        guard.setattr("builtins.open", reject_file_access)
        guard.setattr("io.open", reject_file_access)
        guard.setattr("os.open", reject_file_access)
        guard.setattr(np, "load", reject_file_access)
        guard.setattr(torch, "load", reject_file_access)
        result = model.extract_features(iq)

    assert all(not value.requires_grad and value.grad_fn is None for value in result.values())
    assert iq.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.isfinite(value).all() for value in result.values())


def test_visualization_defaults_and_removed_modes(tmp_path):
    from sgssc_amc.features import add_visualize_parser

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="task", required=True)
    add_visualize_parser(commands, tmp_path)
    args = parser.parse_args(["visualize"])
    assert args.split == "test_unseen"
    assert (args.seed, args.threads, args.max_iter) == (20261031, 4, 1500)
    for unsupported in (["--split", "all"], ["--split", "pink11_unseen"],
                        ["--feature-space", "encoder_signature"],
                        ["--samples-per-class", "50"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["visualize", *unsupported])


def test_classifier_pca_uses_every_l2_row_and_original_tsne_settings(monkeypatch):
    pytest.importorskip("sklearn")
    import sklearn.manifold
    from sgssc_amc.features import classifier_tsne

    recorded = {}

    class FakeTSNE:
        def __init__(self, **kwargs):
            recorded["options"] = kwargs
            self.kl_divergence_ = 0.25
            self.n_iter_ = 1499

        def fit_transform(self, reduced):
            recorded["reduced"] = reduced.copy()
            return reduced[:, :2].astype(np.float32)

    monkeypatch.setattr(sklearn.manifold, "TSNE", FakeTSNE)
    values = np.random.default_rng(20261031).normal(size=(64, 128)).astype(np.float32)
    values[0] = 0  # The documented epsilon must handle a zero feature row.
    original = values.copy()
    coordinates, pca, metadata = classifier_tsne(values, threads=1)
    normalized = values.astype(np.float64)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    assert pca.n_samples_ == len(values)
    assert pca.n_components_ == 50 and pca.svd_solver == "full" and pca.whiten is False
    np.testing.assert_allclose(pca.mean_, normalized.mean(axis=0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(recorded["reduced"], pca.transform(normalized), rtol=1e-11, atol=1e-13)
    np.testing.assert_array_equal(coordinates, recorded["reduced"][:, :2].astype(np.float32))
    np.testing.assert_array_equal(values, original)
    assert recorded["options"] == dict(n_components=2, perplexity=30.0,
        early_exaggeration=12.0, learning_rate="auto", max_iter=1500, init="pca",
        metric="euclidean", method="barnes_hut", angle=0.5,
        random_state=20261031, n_jobs=1)
    assert metadata["windows"] == 64
    assert metadata["pca"]["explained_variance_ratio_sum"] == pytest.approx(pca.explained_variance_ratio_.sum())
    assert metadata["tsne"]["kl_divergence"] == 0.25
    assert set(inspect.signature(classifier_tsne).parameters) == {"features", "seed", "threads", "max_iter"}


@pytest.mark.parametrize("values", [np.zeros((64, 128)), np.ones((64, 128)),
                                    np.full((64, 128), np.nan),
                                    np.ones((49, 128)), np.ones((64, 32))])
def test_classifier_tsne_rejects_invalid_or_constant_geometry(values):
    pytest.importorskip("sklearn")
    from sgssc_amc.features import classifier_tsne

    with pytest.raises(ValueError):
        classifier_tsne(values, threads=1)


@pytest.mark.parametrize("split", ["test_seen", "test_unseen"])
def test_visualization_reads_only_one_split_and_preserves_every_row(tmp_path, monkeypatch, split):
    pytest.importorskip("sklearn")
    from sgssc_amc import features

    count = 55
    rng = np.random.default_rng(12)
    iq = rng.normal(size=(count, 64, 2)).astype(np.float32)
    iq[:, 0, 0] = np.arange(count)
    ids = np.array([f"{split}-{index}" for index in range(count)])
    labels = np.arange(count, dtype=np.int64) % 11
    vectors = rng.normal(size=(count, 128)).astype(np.float32)
    logits = rng.normal(size=(count, 11)).astype(np.float32)
    coordinates = rng.normal(size=(count, 2)).astype(np.float32)
    data_dir = tmp_path / "data"
    (data_dir / "classification").mkdir(parents=True)
    source = data_dir / "classification" / f"{split}.npz"
    np.savez_compressed(source, iq=iq, labels=labels, base_ids=ids)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"mock checkpoint; no model training or research archive access")
    config = SGSSCConfig(length=64, patch=16, dim=128, depth=2, heads=4, ff=32)

    class FakeModel:
        def __init__(self):
            self.config = config

        def extract_features(self, batch):
            rows = batch[:, 0, 0].to(torch.int64).numpy()
            return {"classifier_features": torch.from_numpy(vectors[rows]),
                    "logits": torch.from_numpy(logits[rows])}

    monkeypatch.setattr(features, "load_checkpoint", lambda path: (FakeModel(), {"schema": "fixture"}))
    fitted = []

    def fake_fit(values, seed, threads, max_iter):
        fitted.append(values.copy())
        pca = SimpleNamespace(mean_=np.zeros(128), components_=np.zeros((50, 128)),
                              explained_variance_=np.ones(50), explained_variance_ratio_=np.full(50, 0.02))
        return coordinates, pca, {"fixture": True}

    monkeypatch.setattr(features, "classifier_tsne", fake_fit)
    real_load = np.load
    read_paths = []

    def guarded_load(path, *args, **kwargs):
        assert path == source, "Feature extraction must read only its selected held-out NPZ"
        read_paths.append(path)
        return real_load(path, *args, **kwargs)

    args = argparse.Namespace(split=split, data_dir=data_dir, checkpoint=checkpoint,
                              output=tmp_path / "output", batch_size=17, threads=1,
                              seed=20261031, max_iter=1500, no_plots=True)
    with monkeypatch.context() as guard:
        guard.setattr(np, "load", guarded_load)
        report = features.run_visualization(args)
    assert read_paths == [source]
    assert len(fitted) == 1
    np.testing.assert_array_equal(fitted[0], vectors)
    with np.load(args.output / split / "features.npz", allow_pickle=False) as result:
        assert set(result.files) == {"classifier_features", "logits", "predictions", "labels", "base_ids", "coordinates"}
        for key, expected in (("classifier_features", vectors), ("logits", logits),
                              ("coordinates", coordinates), ("labels", labels), ("base_ids", ids)):
            np.testing.assert_array_equal(result[key], expected)
    with (args.output / split / "coordinates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["base_id"] for row in rows] == ids.tolist()
    assert {row["split"] for row in rows} == {split}
    assert report["data"]["windows"] == count and report["training_updates"] == 0
    assert report["added_noise"] is False
    assert not (args.output / ("test_seen" if split == "test_unseen" else "test_unseen")).exists()


def test_feature_plot_keeps_all_windows_and_has_only_two_black_legend_labels(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.colors import to_rgba
    from sgssc_amc.feature_plots import write_feature_plots

    labels = np.repeat(np.arange(11), 200)
    coordinates = np.random.default_rng(1).normal(size=(2200, 2))
    original = coordinates.copy()
    captured = []

    def check_figure(figure, path, **kwargs):
        figure.canvas.draw()
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_aspect() == 1.0
        assert axis.get_title() == "SGSSC classifier features\nNo added noise"
        assert [len(collection.get_offsets()) for collection in axis.collections] == [1800, 200, 200]
        np.testing.assert_array_equal(axis.collections[1].get_offsets(), coordinates[labels == 8])
        np.testing.assert_array_equal(axis.collections[2].get_offsets(), coordinates[labels == 9])
        np.testing.assert_allclose(axis.collections[1].get_facecolor()[0, :3], to_rgba("#0072B2")[:3])
        np.testing.assert_allclose(axis.collections[2].get_facecolor()[0, :3], to_rgba("#D55E00")[:3])
        assert [text.get_text() for text in figure.legends[0].get_texts()] == ["AM-DSB", "AM-SSB"]
        for text in figure.findobj(Text):
            if text.get_visible() and text.get_text():
                assert to_rgba(text.get_color()) == to_rgba("black")
        assert not figure.texts  # No subtitle/footer outside the requested black title.
        captured.append(path.name)

    monkeypatch.setattr(Figure, "savefig", check_figure)
    artifacts = write_feature_plots(coordinates, labels, tmp_path)
    assert set(captured) == set(artifacts.values()) == {"feature_tsne.png", "feature_tsne.svg"}
    np.testing.assert_array_equal(coordinates, original)


