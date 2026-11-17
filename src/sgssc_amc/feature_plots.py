"""One fixed 2D classifier-feature embedding, with AM-DSB and AM-SSB highlighted.

This module only renders supplied coordinates. It neither fits an embedding
nor reads model or signal data. All supplied points remain in the figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


HIGHLIGHTS = ((8, "AM-DSB", "#0072B2", "o"), (9, "AM-SSB", "#D55E00", "^"))


def write_feature_plots(coordinates, labels, output: Path) -> dict[str, str]:
    """Write a single-panel PNG and SVG; return filenames relative to output.

    True class IDs 8/9 use blue circles/orange triangles. The other nine
    classes remain as gray context points without a legend entry. Labels do
    not change coordinates, axis orientation, sample counts, or selection.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib import rc_context

    coordinates, labels = np.asarray(coordinates), np.asarray(labels)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates)
            or not np.isfinite(coordinates).all()):
        raise ValueError("coordinates must be a nonempty finite [N, 2] array")
    if (labels.shape != (len(coordinates),) or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels > 10))):
        raise ValueError("labels must contain N integer class IDs in [0, 10]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {"static": "feature_tsne.png", "vector": "feature_tsne.svg"}
    with rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "text.color": "black", "axes.labelcolor": "black",
                     "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black"}):
        figure = Figure(figsize=(6.8, 7.2), facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        figure.subplots_adjust(left=0.13, right=0.96, bottom=0.16, top=0.85)
        background = ~np.isin(labels, (8, 9))
        axis.scatter(coordinates[background, 0], coordinates[background, 1],
                     c="#BEC3C9", s=11, alpha=0.42, linewidths=0, zorder=1)
        handles = []
        for class_id, name, color, marker in HIGHLIGHTS:
            selected = labels == class_id
            axis.scatter(coordinates[selected, 0], coordinates[selected, 1],
                         c=color, marker=marker, s=24, alpha=0.86,
                         linewidths=0.2, edgecolors="white", zorder=2)
            handles.append(Line2D([], [], linestyle="", marker=marker,
                                  markerfacecolor=color, markeredgecolor="white",
                                  markersize=8, label=name))
        center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
        half_span = max(float(np.ptp(coordinates, axis=0).max()), 1.0) * 0.55
        axis.set_xlim(center[0] - half_span, center[0] + half_span)
        axis.set_ylim(center[1] - half_span, center[1] + half_span)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("SGSSC classifier features\nNo added noise", color="black", fontsize=14, pad=12)
        axis.set_xlabel("t-SNE 1", color="black", fontsize=11)
        axis.set_ylabel("t-SNE 2", color="black", fontsize=11)
        axis.tick_params(colors="black", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.7)
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
                      ncol=2, frameon=False, fontsize=11, labelcolor="black")
        for filename in files.values():
            figure.savefig(output / filename, dpi=220, facecolor="white")
    return files
