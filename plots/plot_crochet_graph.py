import numpy as np
import trimesh as tm
import pyvista as pv


def plot_crochet_graph(
    row_graph_edges: np.ndarray,
    col_graph_edges: np.ndarray,
    notebook: bool = True,
):
    """Plot the crochet graph row and column edges in 3D.

    Draws row edges in red and column edges in blue as line segments in 3D
    space, reflecting the underlying graph structure used to generate the
    crochet pattern.

    Args:
        row_graph_edges: Row-direction edges, shape (E, 2, 3). Each entry is a
            pair of 3-D endpoint coordinates, as returned by ``get_crochet_graph``.
        col_graph_edges: Column-direction edges, shape (E, 2, 3). Same format.
        notebook: Pass True when running inside a Jupyter notebook so PyVista
            uses its inline renderer; False opens a standalone window.
    """
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_lines(row_graph_edges.reshape((-1, 3)), color="red", width=5, label="Row Edges", connected=False)
    plotter.add_lines(col_graph_edges.reshape((-1, 3)), color="Blue", width=5, label="Column Edges", connected=False)
    plotter.add_legend()
    plotter.show()


def plot_flat_mesh(
    flat_mesh: tm.Trimesh,
    sample_points: np.ndarray,
    notebook: bool = True,
):
    """Plot the UV-flattened mesh and the sampled stitch grid in 2D.

    Displays the mesh projected into the UV plane (row_order as X, column_order
    as Y) in cyan with visible edges, and overlays the filtered sample points in
    red. The camera is locked to the XY plane.

    Args:
        flat_mesh: Trimesh whose vertex positions are (row_order, column_order, 0),
            as constructed from ``compute_row_column_order`` output.
        sample_points: (M, 3) array of sample grid points that lie within the
            mesh (already filtered by proximity distance), shape (M, 3).
        notebook: Pass True when running inside a Jupyter notebook so PyVista
            uses its inline renderer; False opens a standalone window.
    """
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(pv.wrap(flat_mesh), color='cyan', label="Flattened Mesh", show_edges=True)
    plotter.add_points(sample_points, color='red', label="Sampled Points")
    plotter.add_legend()
    plotter.view_xy()
    plotter.show()
