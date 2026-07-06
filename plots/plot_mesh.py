import trimesh as tm
import pyvista as pv


def plot_mesh_picker(mesh: tm.Trimesh, notebook: bool = True) -> list[int]:
    """Display an interactive 3D mesh and let the user pick seed vertices.

    Opens a PyVista window showing the mesh in skyblue. Each click on a vertex
    appends its index to the returned list. The list is live — it grows as the
    user picks points, so it can be read after ``plotter.show()`` returns.

    Args:
        mesh: Surface mesh to display and pick from.
        notebook: Pass True when running inside a Jupyter notebook so PyVista
            uses its inline renderer; False opens a standalone window.

    Returns:
        List of picked vertex indices in the order they were selected.
    """
    picked_vertex_ids = []
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(pv.wrap(mesh), color="skyblue", pickable=True)
    plotter.enable_point_picking(
        callback=lambda _, picker: picked_vertex_ids.append(picker.GetPointId()),
        show_message=False,
        use_picker=True,
        color='red',
    )
    plotter.show()
    return picked_vertex_ids
