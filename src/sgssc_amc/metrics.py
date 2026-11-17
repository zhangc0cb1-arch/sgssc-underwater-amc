"""Metrics use explicit definitions and float64 accumulation."""
from __future__ import annotations

import numpy as np


def classification_metrics(labels: np.ndarray, predictions: np.ndarray,
                           class_names: tuple[str, ...] | list[str]) -> dict:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    classes = len(class_names)
    if (labels.ndim != 1 or labels.shape != predictions.shape or not len(labels)
            or labels.dtype.kind not in "iu" or predictions.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= classes))
            or np.any((predictions < 0) | (predictions >= classes))):
        raise ValueError("Invalid labels or predictions")
    confusion = np.bincount(classes * labels + predictions,
                            minlength=classes * classes).reshape(classes, classes)
    support, assigned = confusion.sum(axis=1), confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(true_positive, assigned, out=np.zeros(classes, dtype=float), where=assigned > 0)
    recall = np.divide(true_positive, support, out=np.zeros(classes, dtype=float), where=support > 0)
    f1 = np.divide(2 * true_positive, support + assigned,
                    out=np.zeros(classes, dtype=float), where=(support + assigned) > 0)
    return {"observations": len(labels), "correct": int(true_positive.sum()),
            "accuracy": float(true_positive.sum() / len(labels)), "macro_f1": float(f1.mean()),
            "macro_f1_definition": "unweighted mean over all published classes; undefined per-class F1 is zero",
            "class_names": list(class_names), "confusion_matrix": confusion.tolist(),
            "confusion_axes": {"rows": "true class", "columns": "predicted class"},
            "per_class": [{"class_id": i, "class_name": name, "support": int(support[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, name in enumerate(class_names)]}


def shifted_spectrum(iq: np.ndarray) -> np.ndarray:
    """Complex orthonormal FFT, represented as real/imaginary coordinates."""
    iq = np.asarray(iq)
    if iq.ndim != 3 or iq.shape[-1] != 2:
        raise ValueError("Expected [N, L, 2] I/Q")
    values = iq[..., 0].astype(np.float64) + 1j * iq[..., 1].astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft(values, axis=1, norm="ortho"), axes=1)
    return np.stack((spectrum.real, spectrum.imag), axis=-1)


def _error(prediction: np.ndarray, target: np.ndarray) -> dict:
    squared_error = np.square(prediction - target).sum(dtype=np.float64)
    energy = np.square(target).sum(dtype=np.float64)
    return {"mse": float(squared_error / target.size),
            "nmse": float(squared_error / energy) if energy > 0 else None,
            "target_mean_square": float(energy / target.size)}


def spectrum_metrics(prediction: np.ndarray, target: np.ndarray,
                     masked_iq: np.ndarray) -> dict:
    prediction, target, masked_iq = map(np.asarray, (prediction, target, masked_iq))
    if (prediction.ndim != 3 or prediction.shape[-1] != 2 or prediction.shape != target.shape
            or target.shape != masked_iq.shape or not len(target) or target.shape[1] < 2
            or not all(np.isfinite(a).all() for a in (prediction, target, masked_iq))):
        raise ValueError("Expected finite and matching [N, L, 2] arrays")
    prediction, target = prediction.astype(np.float64), target.astype(np.float64)
    direct = shifted_spectrum(masked_iq)
    non_dc = np.arange(target.shape[1]) != target.shape[1] // 2
    rows = {}
    for name, output in (("model", prediction), ("zero", np.zeros_like(target)),
                         ("masked_input_fft", direct)):
        rows[name] = {**_error(output, target),
                      "non_dc": _error(output[:, non_dc], target[:, non_dc])}
    return {"observations": len(target), "length": target.shape[1], "results": rows,
            "definitions": {"mse": "mean squared error over all samples, frequency bins and real/imaginary coordinates",
                            "nmse": "global squared-error sum divided by global target-energy sum; not mean per-window NMSE",
                            "non_dc": "same metrics with the fftshift DC bin at index L//2 removed",
                            "masked_input_fft": "orthonormal shifted FFT of the masked input without learned recovery",
                            "undefined_nmse": "null when target energy is zero"}}

