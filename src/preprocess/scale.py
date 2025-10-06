import trimesh as tm


def set_surface_area_to_one(mesh: tm.Trimesh) -> tm.Trimesh:
    """ Scales a mesh to have a surface area of 1. """
    mesh.apply_scale(mesh.area)
    return mesh
