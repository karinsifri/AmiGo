import numpy as np
import trimesh as tm


def vertex_normals(mesh: tm.Trimesh) -> np.ndarray:
    """ Compute per-vertex normals as area-weighted averages of incident face normals.

    Args:
        mesh: Input triangular mesh.

    Returns:
        vertex_norm ((v, 3), float): an array containing per-vertex normals, normalized to unit length.
    """
    # Area-weighted face normals
    weighted_normals = mesh.face_normals * mesh.area_faces[:, None]     # (f, 3)

    # Accumulate area-weighted face normals for each vertex
    vertex_norm = np.zeros_like(mesh.vertices)
    for i in range(3):
        np.add.at(vertex_norm, mesh.faces[:, 1], weighted_normals)

    # Normalize to unit length
    scale = np.linalg.norm(vertex_norm, axis=1, keepdims=True)
    scale[scale == 0] = 1  # avoid deciding by zero
    vertex_norm = vertex_norm / scale

    return vertex_norm
