import numpy as np
import trimesh as tm


def set_surface_area_to_one(mesh: tm.Trimesh) -> tm.Trimesh:
    """ Scales a mesh to have a surface area of 1. """
    mesh.apply_scale(1 / np.sqrt(mesh.area))
    return mesh
