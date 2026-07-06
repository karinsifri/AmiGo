import trimesh as tm
import pyvista as pv


def plot_mesh_picker(mesh: tm.Trimesh, notebook: bool = True) -> list[int]:
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
