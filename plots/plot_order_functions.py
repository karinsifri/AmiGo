import numpy as np
import pyvista as pv
import trimesh as tm


def plot_row_column_order(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                          geodesic_path: np.ndarray | None = None) -> tuple[pv.Plotter, pv.Plotter]:
    """Plot row-order and column-order scalar fields on a mesh.

    Args:
        mesh: Surface mesh on which both scalar fields are defined.
        row_order ((v, ), float): Per-vertex row-order values (geodesic distance field).
        column_order ((v, ), float): Per-vertex column-order values (tangent field).
        geodesic_path ((N, 3), float): Optional array of 3-D points along the geodesic path.

    Returns:
        Tuple of (row_plotter, column_plotter).
    """

    def add_common_attributes(plotter: pv.Plotter) -> pv.Plotter:
        """ An internal function that adds common attributes to both of the plotters """
        plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmin(row_order)]), color="red",
                         label="Origin Point")
        plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(row_order)]), color="blue",
                         label="Maxima")
        if geodesic_path is not None:
            plotter.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
        plotter.add_legend()
        return plotter

    row_plotter = pv.Plotter()
    row_plotter.add_mesh(pv.wrap(mesh), scalars=row_order, cmap="jet", label="Row-Order function (f)")
    row_plotter.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=row_order), color="black", line_width=2,
                         label="Equality Lines")
    row_plotter = add_common_attributes(row_plotter)

    column_plotter = pv.Plotter()
    column_plotter.add_mesh(pv.wrap(mesh), scalars=column_order, cmap="jet", label="Column-Order function (g)")
    column_plotter.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=column_order), color="black", line_width=2,
                            label="Equality Lines")
    column_plotter = add_common_attributes(column_plotter)

    return row_plotter, column_plotter
