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
    """Plot a geodesic distance field on a 3D mesh.

    Colors the mesh by distance from the seed vertex using the jet colormap,
    marks the seed (red) and the farthest point (blue) as spheres, draws 20
    isosurface contour lines, and optionally overlays the geodesic path between
    the two points.

    Args:
        mesh: Surface mesh on which the distance field is defined.
        distance_field: Per-vertex geodesic distances, shape (V,).
        seed_vertex_id: Index of the source vertex (shown as the red sphere).
        geodesic_path: Optional (N, 3) array of 3-D points along the geodesic
            path from seed to the farthest vertex. Shown in green when provided.
        notebook: Pass True when running inside a Jupyter notebook so PyVista
            uses its inline renderer; False opens a standalone window.
    """
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
    """Plot the column-order scalar field on the cut mesh.

    Colors the cut mesh by column-order value (v-axis of the UV parameterization)
    using the jet colormap, marks the origin (red) and maxima (blue) as spheres,
    draws 20 isosurface contour lines, and optionally overlays the geodesic path.

    Args:
        cut_mesh: The mesh after the geodesic cut, on which column_order is defined.
        column_order: Per-vertex column-order values, shape (V,).
        origin_point: 3-D coordinates of the seed/origin vertex (red sphere),
            typically ``mesh.vertices[seed_vertex_id]``.
        maxima_point: 3-D coordinates of the farthest vertex (blue sphere),
            typically ``mesh.vertices[np.argmax(distance_field)]``.
        geodesic_path: Optional (N, 3) array of 3-D points along the geodesic
            path. Shown in green when provided.
        notebook: Pass True when running inside a Jupyter notebook so PyVista
            uses its inline renderer; False opens a standalone window.
    """
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
        notebook: Pass True when running inside a Jupyter notebook so PyVista
            uses its inline renderer; False opens a standalone window.
    """
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