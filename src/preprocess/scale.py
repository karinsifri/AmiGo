import numpy as np
import trimesh as tm


def set_surface_area_to_one(mesh: tm.Trimesh) -> tm.Trimesh:
    """ Scales a copy of the mesh to have a surface area of 1.

    Args:
        mesh: Input mesh to scale.

    Returns:
        A new mesh with surface area normalized to 1.

    Raises:
        ValueError: If the mesh has invalid surface area.
    """
    area = float(mesh.area)
    if not np.isfinite(area) or area <= 1e-12:
        raise ValueError(f"Invalid surface area: {area}")
    scaled = mesh.copy()
    scaled.apply_scale(1.0 / np.sqrt(area))
    return scaled
