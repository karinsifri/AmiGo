import numpy as np
import pyvista as pv
import trimesh as tm


def plot_row_column_order(
        mesh: tm.Trimesh,
        row_order: np.ndarray,
        column_order: np.ndarray,
        geodesic_path: np.ndarray | None = None,
):
    """Plot row-order and column-order scalar fields side-by-side on a mesh.

    Opens two sequential PyVista windows (or inline widgets in a notebook):
    the first colored by row_order (u-axis), the second by column_order (v-axis).
    Both windows share the same origin/maxima sphere markers derived from
    row_order extrema and the same optional geodesic path overlay.

    Args:
        mesh: Surface mesh on which both scalar fields are defined.
        row_order: Per-vertex row-order values (geodesic distance field), shape (V,).
        column_order: Per-vertex column-order values (tangent field), shape (V,).
        geodesic_path: Optional (N, 3) array of 3-D points along the geodesic
            path. Shown in green in both windows when provided.

    Returns:
        Tuple of (row_plotter, column_plotter). Call ``plotter.show()`` on each
        to open the windows.
    """
    row_plotter = pv.Plotter()
    row_plotter.add_mesh(pv.wrap(mesh), scalars=row_order, cmap="jet", label="Distance Field")
    row_plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmin(row_order)]), color="red",
                         label="Origin Point")
    row_plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(row_order)]), color="blue",
                         label="Maxima")
    row_plotter.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=row_order), color="black", line_width=2,
                         label="Equality Lines")
    if geodesic_path is not None:
        row_plotter.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    row_plotter.add_legend()

    column_plotter = pv.Plotter()
    column_plotter.add_mesh(pv.wrap(mesh), scalars=column_order, cmap="jet", label="Distance Field")
    column_plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmin(row_order)]), color="red",
                            label="Origin Point")
    column_plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(row_order)]), color="blue",
                            label="Maxima")
    column_plotter.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=column_order), color="black", line_width=2,
                            label="Equality Lines")
    if geodesic_path is not None:
        column_plotter.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    column_plotter.add_legend()
    return row_plotter, column_plotter
