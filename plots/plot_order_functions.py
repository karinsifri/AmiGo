import numpy as np
import trimesh as tm
import pyvista as pv

from src.order_functions import compute_row_column_order


def plot_geodesic_distance_field(
    mesh: tm.Trimesh,
    distance_field: np.ndarray,
    seed_vertex_id: int,
    geodesic_path: np.ndarray | None = None,
    notebook: bool = True,
):
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(pv.wrap(mesh), scalars=distance_field, cmap="jet", label="Distance Field")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[seed_vertex_id]), color="red", label="Origin Point")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(distance_field)]), color="blue", label="Maxima")
    plotter.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=distance_field), color="black", line_width=2, label="Equality Lines")
    if geodesic_path is not None:
        plotter.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    plotter.add_legend()
    plotter.show()


def plot_column_order(
    cut_mesh: tm.Trimesh,
    column_order: np.ndarray,
    origin_point: np.ndarray,
    maxima_point: np.ndarray,
    geodesic_path: np.ndarray | None = None,
    notebook: bool = True,
):
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(pv.wrap(cut_mesh), scalars=column_order, cmap="jet", label="Distance Field")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=origin_point), color="red", label="Origin Point")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=maxima_point), color="blue", label="Maxima")
    plotter.add_mesh(pv.wrap(cut_mesh).contour(isosurfaces=20, scalars=column_order), color="black", line_width=2, label="Equality Lines")
    if geodesic_path is not None:
        plotter.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    plotter.add_legend()
    plotter.show()


def plot_row_column_order(
    mesh: tm.Trimesh,
    row_order: np.ndarray,
    column_order: np.ndarray,
    geodesic_path: np.ndarray | None = None,
    notebook: bool = False,
):
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(pv.wrap(mesh), scalars=row_order, cmap="jet", label="Distance Field")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmin(row_order)]), color="red", label="Origin Point")
    plotter.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(row_order)]), color="blue", label="Maxima")
    plotter.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=row_order), color="black", line_width=2, label="Equality Lines")
    if geodesic_path is not None:
        plotter.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    plotter.add_legend()
    plotter.show(interactive_update=True, auto_close=False)

    plotter2 = pv.Plotter(notebook=notebook)
    plotter2.add_mesh(pv.wrap(mesh), scalars=column_order, cmap="jet", label="Distance Field")
    plotter2.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmin(row_order)]), color="red", label="Origin Point")
    plotter2.add_mesh(pv.Sphere(radius=0.007, center=mesh.vertices[np.argmax(row_order)]), color="blue", label="Maxima")
    plotter2.add_mesh(pv.wrap(mesh).contour(isosurfaces=20, scalars=column_order), color="black", line_width=2, label="Equality Lines")
    if geodesic_path is not None:
        plotter2.add_lines(geodesic_path, color="green", width=7, label="Geodesic Path", connected=True)
    plotter2.add_legend()
    plotter2.show(interactive_update=True, auto_close=False)