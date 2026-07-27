import numpy as np
import trimesh as tm
import pyvista as pv

from src.consts import EPSILON
from src.crochet_graph import CrochetGraph


def plot_crochet_graph(graph: CrochetGraph) -> pv.Plotter:
    """Plot the crochet graph edges and crease

    Args:
        graph: Fully constructed crochet graph.

    Returns:
        Configured plotter
    """
    blo: np.ndarray[np.bool_] = (graph.creases == 1)
    flo: np.ndarray[np.bool_] = (graph.creases == -1)
    blo_col = blo[graph.column_edges[:, 0]]
    flo_col = flo[graph.column_edges[:, 0]]

    plotter = pv.Plotter()
    plotter.add_lines(graph.vertices[graph.row_edges].reshape((-1, 3)), color="red", width=5, label="Row Edges",
                      connected=False)
    plotter.add_lines(graph.vertices[graph.column_edges].reshape((-1, 3)), color="blue", width=5, label="Column Edges",
                      connected=False)
    if blo_col.any():
        plotter.add_lines(graph.vertices[graph.column_edges[blo_col]].reshape((-1, 3)), color="green", width=6,
                          label="BLO Edges", connected=False)
    if flo_col.any():
        plotter.add_lines(graph.vertices[graph.column_edges[flo_col]].reshape((-1, 3)), color="cyan", width=6,
                          label="FLO Edges", connected=False)
    plotter.add_legend()
    return plotter


def plot_flat_mesh(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray, stitch_size: float) -> pv.Plotter:
    """Plot the UV-flattened mesh and the sampled stitch grid in 2D.

    Args:
        mesh: Cut mesh on which row_order and column_order are defined on
        row_order ((v, ), float): Per-vertex row-order values (u-axis)
        column_order((v, ), float): Per-vertex column-order values (v-axis)
        stitch_size: Spacing between sampled stitch points along both axes

    Returns:
        Configured plotter
    """
    flat_mesh = tm.Trimesh(np.vstack([column_order, row_order, np.zeros_like(row_order)]).T, mesh.faces)

    sample_col_values, sample_row_values = np.meshgrid(np.arange(0, column_order.max(), stitch_size),
                                                       np.arange(0, row_order.max(), stitch_size))
    sample_points = np.stack((np.append(sample_col_values, 0), np.append(sample_row_values, row_order.max()),
                              np.zeros((sample_row_values.size + 1,))), axis=-1)
    _, point_distances, _ = tm.proximity.closest_point(flat_mesh, sample_points)
    sample_points = sample_points[point_distances < EPSILON]

    plotter = pv.Plotter()
    plotter.add_mesh(pv.wrap(flat_mesh), color='cyan', label="Flattened Mesh", show_edges=True)
    plotter.add_points(sample_points, color='red', label="Sampled Points")
    plotter.add_legend()
    plotter.view_xy()
    return plotter
