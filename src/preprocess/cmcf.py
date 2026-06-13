import gpytoolbox as gpy
import numpy as np
import scipy.sparse as sp
import trimesh as tm
from potpourri3d import compute_distance_multisource

from shapeop import shape_operator_ftf


def smooth_craters(mesh: tm.Trimesh) -> tm.Trimesh:
    tstep = mesh.edges_unique_length.mean() ** 2

    W = - gpy.cotangent_laplacian(mesh.vertices, mesh.faces)

    for i in range(250):
        dminf, dmaxf, kminf, kmaxf = shape_operator_ftf(mesh)
        HhfaceSO = 0.5 * (kminf + kmaxf)  # mean curvature on faces
        GgfaceSO = kminf * kmaxf  # gaussian curvature on faces

        area_to_change = (GgfaceSO >= 0) & (HhfaceSO <= 0)
        vertices_to_change = np.unique(mesh.faces[np.nonzero(area_to_change)])

        if len(vertices_to_change) == 0:
            break

        ff = compute_distance_multisource(mesh.vertices, mesh.faces, vertices_to_change)
        sigma = 3 * mesh.edges_unique_length.mean()
        gaussianff = np.exp(-ff / sigma)

        M = sp.spdiags(vertex_areas(mesh), 0, len(mesh.vertices), len(mesh.vertices))
        V_ = sp.linalg.spsolve(sp.csc_matrix(M - tstep * sp.diags(gaussianff) @ W), M @ mesh.vertices)

        m_ = tm.Trimesh(V_, mesh.faces)

        mesh = m_

    return mesh


""" helper functions """


def vertex_areas(mesh: tm.Trimesh) -> np.ndarray:
    """
    Compute vertex areas from mass matrix.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        A trimesh object

    Returns:
    --------
    va : numpy.ndarray
        Vertex areas
    """
    M = gpy.massmatrix(mesh.vertices, mesh.faces, 'full')
    va = np.array(M.sum(axis=1)).flatten()
    return va

