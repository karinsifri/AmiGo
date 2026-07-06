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
    fig, ax = plt.subplots()
    ax.text(0.1, 0.5, instructions_text, ha='left', va='center', fontsize=24, transform=ax.transAxes)
    ax.axis('off')
    return fig, ax
