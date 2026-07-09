from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure


def plot_stitch_rows(row_stitch_connectivity, annotations: list[str], split_creases: Optional[list[np.ndarray]] = None,
                     figsize: tuple[int, int] = (10, 15), ) -> tuple[Figure, Axes]:
    """Plot per-row stitch connectivity as a 2D diagram with row instructions annotations.

    Args:
        row_stitch_connectivity: Sequence of per-row arrays, each of shape (S, 2) where each row gives connectivity
            information between two consecutive rows.
        annotations: One label string per row, displayed to the right of the row.  Length must match
            ``row_stitch_connectivity``.
        split_creases: Optional per-row crease labels.
        figsize: Matplotlib figure size as (width, height) in inches.

    Returns:
        The created ``(Figure, Axes)`` pair.
    """
    fig, ax = plt.subplots(figsize=figsize)
    right_arrow = (0, 1)
    left_arrow = (1, 0)
    for row_idx, (row_connectivity, label) in enumerate(zip(row_stitch_connectivity, annotations, strict=True)):
        stitch_segments = np.stack(
            [row_connectivity, np.repeat([[row_idx - 0.25, row_idx + 0.25]], len(row_connectivity), axis=0)],
            axis=-1,
        )
        ax.add_collection(LineCollection(stitch_segments, colors='b', linewidths=2))
        if split_creases is not None:
            segment_creases = split_creases[row_idx][row_connectivity[:, 0]]
            blo = segment_creases == 1
            flo = segment_creases == -1
            if blo.any():
                ax.add_collection(LineCollection(stitch_segments[blo], colors='green', linewidths=2))
            if flo.any():
                ax.add_collection(LineCollection(stitch_segments[flo], colors='cyan', linewidths=2))
        row_extent = row_connectivity[:, 0].max()
        ax.hlines(row_idx - 0.25, 0, row_extent, color='r')
        ax.hlines(row_idx + 0.25, 0, row_connectivity[:, 1].max(), color='r')
        ax.annotate(label, xy=[row_connectivity[-1, :].max() + 1, row_idx])

        arrow_y = row_idx - 0.4
        arrow_start, arrow_end = right_arrow if row_idx % 2 == 0 else left_arrow
        ax.annotate('', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
    ax.set_ylabel("row")
    for spine in ['right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim(-0.5, len(row_stitch_connectivity) - 0.5)
    return fig, ax


def plot_final_instructions(instructions_text: str) -> tuple[Figure, Axes]:
    """Display formatted crochet instructions as a text-only figure.

    Args:
        instructions_text: The full instruction string to display.

    Returns:
        The created ``(Figure, Axes)`` pair.
    """
    fig, ax = plt.subplots()
    ax.text(0.1, 0.5, instructions_text, ha='left', va='center', fontsize=24, transform=ax.transAxes)
    ax.axis('off')
    return fig, ax
