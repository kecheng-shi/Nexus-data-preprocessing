"""Plotting utilities used across Nexus notebooks."""

from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = ["segments_from_labels", "plot_phase_heatmaps"]


def segments_from_labels(dates: pd.Series, labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Convert a label series into contiguous (start, end, label) spans."""
    segs: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if len(dates) == 0:
        return segs
    start = dates.iloc[0]
    cur = labels.iloc[0]
    for d, lab in zip(dates.iloc[1:], labels.iloc[1:]):
        if lab != cur:
            segs.append((pd.to_datetime(start), pd.to_datetime(d), str(cur)))
            start, cur = d, lab
    segs.append((pd.to_datetime(start), pd.to_datetime(dates.iloc[-1]), str(cur)))
    return segs


def plot_phase_heatmaps(
    matrix_map: dict[str, pd.DataFrame] | OrderedDict[str, pd.DataFrame],
    cmap: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    cbar_label: str = "",
    title_prefix: str = "",
    x_label: str = "Asset class",
    y_label: str = "Macro factor",
) -> None:
    """Render a panel of heatmaps keyed by phase name."""

    if not matrix_map:
        print("No phase-specific matrices to plot.")
        return

    phase_count = len(matrix_map)
    max_cols = max((mat.shape[1] for mat in matrix_map.values()), default=1)
    max_rows = max((mat.shape[0] for mat in matrix_map.values()), default=1)
    fig_width = max(12, 1.2 * max_cols)
    fig_height = max(6, 0.6 * max_rows) * max(1, phase_count)

    fig, axes = plt.subplots(
        phase_count,
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (phase_name, matrix) in zip(axes_flat, matrix_map.items()):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=False,
            cbar_kws={"label": cbar_label} if cbar_label else None,
        )
        ax.set_title(f"{title_prefix}{phase_name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    for ax in axes_flat[len(matrix_map):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()

