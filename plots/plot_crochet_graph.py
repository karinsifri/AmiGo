import numpy as np
import trimesh as tm
import pyvista as pv


def plot_crochet_graph(
    row_graph_edges: np.ndarray,
    col_graph_edges: np.ndarray,
    notebook: bool = True,
):
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
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(pv.wrap(flat_mesh), color='cyan', label="Flattened Mesh", show_edges=True)
    plotter.add_points(sample_points, color='red', label="Sampled Points")
    plotter.add_legend()
    plotter.view_xy()
    plotter.show()
