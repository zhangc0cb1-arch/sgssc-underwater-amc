"""Load the tensor-only public checkpoint without original project dependencies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models import SGSSCConfig, SGSSCEncoder, SGSSCReadout, FlattenSpectralHead

CLASS_NAMES = ("BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64", "GFSK",
               "CPFSK", "AM-DSB", "AM-SSB", "WBFM")


class InferenceModel(nn.Module):
    """One fixed encoder with two independently evaluated task heads."""

    def __init__(self, config: SGSSCConfig):
        super().__init__()
        self.config = config
        self.encoder = SGSSCEncoder(config)
        self.readout = SGSSCReadout(config)
        self.spectral_head = FlattenSpectralHead(config)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # This wrapper is inference-only; training uses the underlying modules.
        super().train(False)
        return self

    @torch.inference_mode()
    def classify(self, iq: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encoder(iq))

    @torch.inference_mode()
    def extract_features(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return both feature levels and logits from one frozen encoder pass.

        ``encoder_tokens`` is the final self-supervised encoder representation
        [B, 32, 128] for the released checkpoint. ``classifier_features`` is
        the [B, 128] pooled representation from the label-supervised downstream
        readout, immediately before its final linear classifier. ``logits``
        has shape [B, 11] and is identical to ``classify(iq)``. All outputs are
        computed in inference mode with the released CPU model in eval mode.
        """
        layers = self.encoder(iq)
        pooled = self.readout.extract_features(layers)
        return {"encoder_tokens": layers[-1], "classifier_features": pooled,
                "logits": self.readout.classifier(pooled)}

    @torch.inference_mode()
    def recover_spectrum(self, masked_iq: torch.Tensor) -> torch.Tensor:
        return self.spectral_head(self.encoder(masked_iq)[-1])


def load_checkpoint(path: str | Path) -> tuple[InferenceModel, dict[str, Any]]:
    """Accept the public schema only; unrestricted pickle loading is never used."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"schema", "model_config", "class_names", "encoder_state",
                "readout_state", "spectral_head_state", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Incomplete public inference checkpoint")
    if payload["schema"] != "sgssc-inference-v1":
        raise ValueError("Unsupported checkpoint schema")
    if tuple(payload["class_names"]) != CLASS_NAMES:
        raise ValueError("Checkpoint class order differs from the published protocol")
    if not isinstance(payload["metadata"], dict):
        raise ValueError("Checkpoint metadata must be a mapping")
    config = SGSSCConfig(**payload["model_config"])
    # Do not change callers' global random stream when constructing the modules.
    with torch.random.fork_rng(devices=[]):
        model = InferenceModel(config)
    for name, module in (("encoder_state", model.encoder),
                         ("readout_state", model.readout),
                         ("spectral_head_state", model.spectral_head)):
        state = payload[name]
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Missing tensor state: {name}")
        if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
               for v in state.values()):
            raise ValueError(f"Nonfinite or non-tensor parameters: {name}")
        module.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, {"schema": payload["schema"], "model_config": payload["model_config"],
                   "class_names": list(CLASS_NAMES), "metadata": payload["metadata"]}

