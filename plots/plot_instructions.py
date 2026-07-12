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
        figsize: Matplotlib figure size as (width, height) in inches, for the stitch-diagram portion; the figure
            is widened afterwards to fit the annotation text.

    Returns:
        The created ``(Figure, Axes)`` pair.
    """
    fig, ax = plt.subplots(figsize=figsize)
    right_arrow = (0, 1)
    left_arrow = (1, 0)
    texts = []
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
        texts.append(ax.annotate(label, xy=[row_connectivity[-1, :].max() + 1, row_idx], annotation_clip=False))

        arrow_y = row_idx - 0.4
        arrow_start, arrow_end = right_arrow if row_idx % 2 == 0 else left_arrow
        ax.annotate('', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
    ax.set_ylabel("row")
    for spine in ['right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim(-0.5, len(row_stitch_connectivity) - 0.5)

    _reserve_right_margin_for_text(fig, ax, texts)
    return fig, ax


def _reserve_right_margin_for_text(fig: Figure, ax: Axes, texts: list, pad_inches: float = 0.3) -> None:
    """Widen the figure so that none of ``texts`` is clipped by the current axes' right edge.

    Each text keeps its own position (e.g. right where its row's line ends); only the blank canvas to the
    right of the axes is enlarged to fit whichever text overflows the most, so nothing else moves.
    """
    if not texts:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_right_px = ax.get_window_extent(renderer=renderer).x1
    max_overflow_in = max((t.get_window_extent(renderer=renderer).x1 - axes_right_px for t in texts),
                          default=0) / fig.dpi
    if max_overflow_in <= 0:
        return

    fig_width_in, fig_height_in = fig.get_size_inches()
    axes_left_in = fig.subplotpars.left * fig_width_in
    axes_right_in = fig.subplotpars.right * fig_width_in
    new_fig_width_in = fig_width_in + max_overflow_in + pad_inches
    fig.set_size_inches(new_fig_width_in, fig_height_in)
    fig.subplots_adjust(left=axes_left_in / new_fig_width_in, right=axes_right_in / new_fig_width_in)


def plot_final_instructions(instructions_text: str, pad_inches: float = 0.3) -> tuple[Figure, Axes]:
    """Display formatted crochet instructions as a text-only figure, sized to fit the text exactly.

    Args:
        instructions_text: The full instruction string to display.
        pad_inches: Padding added around the text on every side.

    Returns:
        The created ``(Figure, Axes)`` pair.
    """
    fig, ax = plt.subplots()
    ax.axis('off')
    text = ax.text(0, 0.5, instructions_text, ha='left', va='center', fontsize=24, transform=ax.transAxes)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer)
    width_in = bbox.width / fig.dpi + 2 * pad_inches
    height_in = bbox.height / fig.dpi + 2 * pad_inches
    fig.set_size_inches(width_in, height_in)

    ax.set_position((0, 0, 1, 1))
    text.set_position((pad_inches / width_in, 0.5))

    return fig, ax
