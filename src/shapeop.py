import numpy as np
import trimesh as tm

EPSILON = 1e-15


def vertex_normals(mesh: tm.Trimesh) -> np.ndarray:
    """ Compute per-vertex normals as area-weighted averages of incident face normals.

    Args:
        mesh: Input triangular mesh.

    Returns:
        vertex_norm ((v, 3), float): an array containing per-vertex normals, normalized to unit length.
    """
    # Area-weighted face normals
    weighted_normals = mesh.face_normals * mesh.area_faces[:, None]  # (f, 3)

    # Accumulate area-weighted face normals for each vertex
    vertex_norm = np.zeros_like(mesh.vertices)
    for i in range(3):
        np.add.at(vertex_norm, mesh.faces[:, i], weighted_normals)

    # Normalize to unit length
    scale = np.linalg.norm(vertex_norm, axis=1, keepdims=True)
    scale[scale == 0] = 1  # avoid deciding by zero
    vertex_norm = vertex_norm / scale

    return vertex_norm


def shape_operator_ftf(mesh: tm.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ita = np.tile(1 / mesh.area_faces, (3, 1)).T

    e0 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]]  # -> across vertex 0 in the face
    e1 = mesh.vertices[mesh.faces[:, 0]] - mesh.vertices[mesh.faces[:, 2]]  # -> across vertex 1 in the face
    e2 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]  # -> across vertex 2 in the face

    ge = [0.5 * ita * np.cross(mesh.face_normals, e, axis=1) for e in [e0, e1, e2]]

    dminf = np.zeros(mesh.faces.shape)
    dmaxf = np.zeros(mesh.faces.shape)
    kminf = np.zeros(len(mesh.faces))
    kmaxf = np.zeros(len(mesh.faces))

    vn = vertex_normals(mesh)

    for face_idx in range(mesh.faces.shape[0]):
        s = np.stack([vn[mesh.faces[face_idx, i]].reshape(-1, 1) @ ge[i][face_idx].reshape(1, -1)
                      for i in range(3)]).sum(axis=0)

        p = np.eye(3) - mesh.face_normals[face_idx].reshape(-1, 1) @ mesh.face_normals[face_idx].reshape(1, -1)

        s = p @ s
        s = (s + s.T) / 2

        d, v = np.linalg.eig(s)
        v = np.real(v)
        d = np.real(d)

        sort_idx = np.argsort(np.abs(d))
        abs_dd = np.abs(d[sort_idx])

        # first condition - all ev ~= 0 or 1 ev = 0
        if np.all(abs_dd > EPSILON) or (abs_dd[0] < EPSILON and np.all(abs_dd[1:] > EPSILON)):
            # eliminate normal eigenvector
            e = np.vstack([e0[face_idx], e1[face_idx]]).T

            min_idx = np.argmin(np.linalg.norm(v.T @ e, axis=-1))

            v = np.delete(v, min_idx, axis=1)  # remove column min_idx
            d = np.delete(d, min_idx)  # remove element min_idx

            s_idx = np.argsort(d)

            v = v[:, s_idx]
            d = d[s_idx]

        else:
            raise RuntimeError(f"did not meet first condition on index {face_idx}")

        dminf[face_idx] = v[:, 0].T
        dmaxf[face_idx] = v[:, 1].T
        kminf[face_idx] = d[0]
        kmaxf[face_idx] = d[1]

    return dminf, dmaxf, kminf, kmaxf
