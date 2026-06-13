import gpytoolbox as gpy
import numpy as np
import scipy.sparse as sp
import trimesh as tm
from potpourri3d import compute_distance_multisource

from src.shapeop import shape_operator_ftf


def smooth_craters(mesh: tm.Trimesh) -> tm.Trimesh:
    """Apply modified mean-curvature flow to concave elliptic (crater) regions.

    Implements the conformalized mean-curvature flow (CMCF) of Kazhdan, Solomon,
    and Ben-Chen (2012) restricted to faces where both principal curvatures are
    non-positive: K = k_min * k_max >= 0 and H = (k_min + k_max) / 2 <= 0.
    This combination identifies bowl-shaped concavities — the surface curves
    inward in every direction.

    Args:
        mesh: Input triangular mesh. Must be a closed orientable manifold.

    Returns:
        A new Trimesh with the same face connectivity as the input but with
        vertex positions modified to smooth out all crater regions.
    """
    time_step = mesh.edges_unique_length.mean() ** 2

    laplacian = -gpy.cotangent_laplacian(mesh.vertices, mesh.faces)

    for _ in range(250):
        _, _, min_curvature_per_face, max_curvature_per_face = shape_operator_ftf(mesh)

        mean_curvature_per_face = 0.5 * (min_curvature_per_face + max_curvature_per_face)
        gaussian_curvature_per_face = min_curvature_per_face * max_curvature_per_face

        crater_face_mask = (gaussian_curvature_per_face >= 0) & (mean_curvature_per_face <= 0)
        crater_vertex_indices = np.unique(mesh.faces[np.nonzero(crater_face_mask)])

        if len(crater_vertex_indices) == 0:
            break

        dist_to_crater = compute_distance_multisource(mesh.vertices, mesh.faces, crater_vertex_indices)

        gaussian_bandwidth = 3 * mesh.edges_unique_length.mean()
        crater_influence = np.exp(-dist_to_crater / gaussian_bandwidth)

        full_mass_matrix = gpy.massmatrix(mesh.vertices, mesh.faces, 'full')
        per_vertex_area = np.array(full_mass_matrix.sum(axis=1)).flatten()
        mass_matrix = sp.spdiags(per_vertex_area, 0, len(mesh.vertices), len(mesh.vertices))

        lhs = sp.csc_matrix(mass_matrix - time_step * sp.diags(crater_influence) @ laplacian)
        rhs = mass_matrix @ mesh.vertices
        new_vertices = sp.linalg.spsolve(lhs, rhs)

        mesh = tm.Trimesh(new_vertices, mesh.faces)

    return mesh
