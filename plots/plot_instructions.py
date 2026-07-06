import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure


def plot_stitch_rows(
    row_stitch_connectivity,
    annotations: list[str],
    figsize: tuple[int, int] = (10, 15),
) -> tuple[Figure, Axes]:
    """Plot per-row stitch connectivity as a 2D diagram.

    Each row is drawn as a set of horizontal blue line segments (one per stitch),
    bounded above and below by red lines marking the row extent. An annotation
    label is placed to the right of each row.

    Suitable for both raw stitch sequences (e.g. ``"dc ch sc tr"``) and folded
    sequences with repetition counts (e.g. ``"3*dc 2*ch (42)"``).

    Args:
        row_stitch_connectivity: Sequence of per-row arrays, each of shape
            (S, 2) where each row gives the [start, end] column positions of
            one stitch, as returned by ``get_row_connectivity``.
        annotations: One label string per row, displayed to the right of the
            row. Length must match ``row_stitch_connectivity``.
        figsize: Matplotlib figure size as (width, height) in inches.

    Returns:
        The created ``(Figure, Axes)`` pair.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for row_idx, (row_connectivity, label) in enumerate(zip(row_stitch_connectivity, annotations)):
        stitch_segments = np.stack(
            [row_connectivity, np.repeat([[row_idx - 0.25, row_idx + 0.25]], len(row_connectivity), axis=0)],
            axis=-1,
        )
        ax.add_collection(LineCollection(stitch_segments, colors='b', linewidths=2))
        ax.hlines(row_idx - 0.25, 0, row_connectivity[:, 0].max(), color='r')
        ax.hlines(row_idx + 0.25, 0, row_connectivity[:, 1].max(), color='r')
        ax.annotate(label, xy=[row_connectivity[-1, :].max() + 1, row_idx])
    ax.set_ylabel("row")
    for spine in ['right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    return fig, ax


def plot_final_instructions(instructions_text: str) -> tuple[Figure, Axes]:
    """Display formatted crochet instructions as a text-only figure.

    Renders the instruction string centred vertically in a plain matplotlib
    figure with all axes hidden, suitable for printing or saving as an image.

    Args:
        instructions_text: The full instruction string to display, as produced
            by ``create_final_instructions``.

    Returns:
        The created ``(Figure, Axes)`` pair.
    """
    fig, ax = plt.subplots()
    ax.text(0.1, 0.5, instructions_text, ha='left', va='center', fontsize=24, transform=ax.transAxes)
    ax.axis('off')
    return fig, ax
