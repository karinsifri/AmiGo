import numpy as np
import trimesh as tm


def set_surface_area_to_one(mesh: tm.Trimesh) -> tm.Trimesh:
    """ Scales a copy of the mesh to have a surface area of 1.

    Args:
        mesh: Input mesh to scale.

    Returns:
        A new mesh with surface area normalized to 1.
    """
    if mesh.area <= 0:
        raise ValueError(f"Mesh has invalid surface area: {mesh.area}. Expected positive value.")
    mesh = mesh.copy()
    mesh.apply_scale(1 / np.sqrt(mesh.area))
    return mesh
