import trimesh as tm
import pyvista as pv


def plot_mesh_picker(mesh: tm.Trimesh) -> list[int]:
    """Display an interactive 3D mesh and let the user pick seed vertices.

    Opens a PyVista window showing the mesh in skyblue. Each click on a vertex
    appends its index to the returned list. The list is live — it grows as the
    user picks points, so it can be read after ``plotter.show()`` returns.

    Args:
        mesh: Surface mesh to display and pick from.

    Returns:
        Tuple of (plotter, picked_vertex_ids). Call ``plotter.show()`` to open
        the window; ``picked_vertex_ids`` is live and grows as the user picks.
    """
    picked_vertex_ids = []
    plotter = pv.Plotter()
    plotter.add_mesh(pv.wrap(mesh), color="skyblue", pickable=True)
    plotter.enable_point_picking(
        callback=lambda _, picker: picked_vertex_ids.append(picker.GetPointId()),
        show_message=False,
        use_picker=True,
        color='red',
    )
    return plotter, picked_vertex_ids
