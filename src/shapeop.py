import numpy as np
import scipy
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
    scale[scale > EPSILON] = 1  # avoid dividing by zero
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
        # Second condition - all ev = 0
        elif np.all(abs_dd < EPSILON):
            v1 = e0[face_idx] / np.linalg.norm(e0[face_idx])
            v2 = np.cross(v1, mesh.face_normals[face_idx])
            v2 = v2 / np.linalg.norm(v2)
            v = np.vstack([v1, v2]).T
            d = [0, 0]
        # Third condition - 2 ev = 0
        elif np.all(abs_dd[:2] < EPSILON) and (abs_dd[-1] > EPSILON):
            v1 = v[:, sort_idx][:, -1]
            dd1 = d[sort_idx][-1]  # different from MATLAB implementation
            assert np.linalg.norm(v1.T @ mesh.face_normals[face_idx]) < EPSILON
            v2 = np.cross(v1.T, mesh.face_normals[face_idx])
            v2 = v2 / np.linalg.norm(v2)
            v = np.vstack([v1, v2]).T
            d = np.array([dd1, 0])
            s_idx = np.argsort(d)
            v = v[:, s_idx]
            d = d[s_idx]
        else:
            raise RuntimeError(f"None of the conditions were met on index {face_idx}")

        dminf[face_idx] = v[:, 0].T
        dmaxf[face_idx] = v[:, 1].T
        kminf[face_idx] = d[0]
        kmaxf[face_idx] = d[1]

    return dminf, dmaxf, kminf, kmaxf


def edge_basis(mesh: tm.Trimesh) -> scipy.sparse.csr_matrix:
    e0 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]]  # -> across vertex 0 in the face
    ne1 = e0 / np.linalg.norm(e0, axis=1, keepdims=True)
    ne2 = np.cross(mesh.face_normals, ne1, axis=1)

    nf = len(mesh.faces)

    # Construct VFI and VFJ exactly as MATLAB does
    i = np.tile(np.arange(nf), 3)
    j = np.arange(nf)
    j = np.concatenate([j, j + nf, j + 2 * nf])

    # Flatten in Fortran order (column-major) to match MATLAB
    b1 = scipy.sparse.csr_matrix((ne1.flatten(order='F'), (i, j)), shape=(nf, 3 * nf))
    b2 = scipy.sparse.csr_matrix((ne2.flatten(order='F'), (i, j)), shape=(nf, 3 * nf))

    # Stack them
    return scipy.sparse.vstack([b1, b2])


def shapeop(mesh: tm.Trimesh) -> scipy.sparse.csr_matrix:
    nf = len(mesh.faces)

    dminf, dmaxf, kminf, kmaxf = shape_operator_ftf(mesh)
    eb = edge_basis(mesh)

    dminfb = (eb @ dminf.flatten(order='F')).reshape(-1, 2, order='F')
    dmaxfb = (eb @ dmaxf.flatten(order='F')).reshape(-1, 2, order='F')

    v11 = dminfb[:, 0]
    v12 = dminfb[:, 1]
    v21 = dmaxfb[:, 0]
    v22 = dmaxfb[:, 1]

    # Create sparse diagonal matrices and stack them into block matrix
    v = scipy.sparse.bmat([
        [scipy.sparse.diags(v11, 0, shape=(nf, nf)), scipy.sparse.diags(v12, 0, shape=(nf, nf))],
        [scipy.sparse.diags(v21, 0, shape=(nf, nf)), scipy.sparse.diags(v22, 0, shape=(nf, nf))]
    ])

    d = scipy.sparse.bmat([
        [scipy.sparse.diags(kminf, 0, shape=(nf, nf)), scipy.sparse.csr_matrix((nf, nf))],
        [scipy.sparse.csr_matrix((nf, nf)), scipy.sparse.diags(kmaxf, 0, shape=(nf, nf))]
    ])

    return v.T @ d @ v

