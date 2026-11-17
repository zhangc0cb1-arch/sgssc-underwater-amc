"""Static figures for the two distinct inference tasks."""
from pathlib import Path
import numpy as np

from .metrics import shifted_spectrum


def confusion_plot(metrics: dict, destination: Path, title: str) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix * 100, support, out=np.zeros_like(matrix), where=support > 0)
    figure = Figure(figsize=(9, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    plot = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    names = metrics["class_names"]
    axis.set_xticks(range(len(names)), names, rotation=50, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted modulation")
    axis.set_ylabel("True modulation")
    axis.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, f"{normalized[i, j]:.1f}", ha="center", va="center", fontsize=7,
                      color="white" if normalized[i, j] > 50 else "#172033")
    figure.colorbar(plot, ax=axis, label="Within-class percentage (%)", shrink=0.8)
    figure.savefig(destination, dpi=150)


def spectrum_plot(prediction: np.ndarray, target: np.ndarray, masked_iq: np.ndarray,
                  base_id: str, destination: Path) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    frequency = np.fft.fftshift(np.fft.fftfreq(len(target)))
    direct = shifted_spectrum(masked_iq[None])[0]
    for coordinate, axis in enumerate(axes):
        axis.plot(frequency, target[:, coordinate], label="Target: FFT of complete I/Q", lw=1.3)
        axis.plot(frequency, prediction[:, coordinate], label="Model recovery", lw=1.0, alpha=0.9)
        axis.plot(frequency, direct[:, coordinate], label="Baseline: FFT of masked I/Q",
                  lw=0.8, alpha=0.6, linestyle="--")
        axis.set_ylabel("Real" if coordinate == 0 else "Imaginary")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Normalized frequency (cycles/sample)")
    figure.suptitle(f"Spectrum recovery: first held-out example ({base_id})\n"
                    "Illustration only; aggregate errors are reported in metrics.json", fontsize=11)
    figure.savefig(destination, dpi=150)

