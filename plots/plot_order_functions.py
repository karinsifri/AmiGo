import numpy as np
import trimesh as tm
import pyvista as pv
from potpourri3d import EdgeFlipGeodesicSolver

from src.order_functions import compute_row_column_order


def plot_row_column_order(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray):
    plotter = pv.Plotter(notebook=False)
    plotter.add_mesh(pv.wrap(mesh), scalars=row_order, cmap="jet", label="Distance Field")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmin(row_order)]), color="red", label="Origin Point")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(row_order)]), color="blue",
                     label="Maxima")
    plotter.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=row_order), color="black", line_width=2,
                     label="Equality Lines")
    # plotter.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    plotter.add_legend()
    plotter.show(interactive_update=True, auto_close=False)

    plotter2 = pv.Plotter(notebook=False)
    plotter2.add_mesh(pv.wrap(cut_mesh), scalars=column_order, cmap="jet", label="Distance Field")
    plotter2.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmin(row_order)]), color="red", label="Origin Point")
    plotter2.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(row_order)]), color="blue",
                     label="Maxima")
    plotter2.add_mesh(pv.wrap(cut_mesh).contour(isosurfaces=20, scalars=column_order), color="black", line_width=2,
                     label="Equality Lines")
    # plotter2.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    plotter2.add_legend()
    plotter2.show(interactive_update=True, auto_close=False)
    pass


if __name__ == '__main__':
    _mesh = tm.load_mesh('../meshes/C_r2.obj')
    seed = 4545
    cut_mesh, _row_order, _column_order = compute_row_column_order(_mesh, seed)
    plot_row_column_order(cut_mesh, _row_order, _column_order)
    pass

