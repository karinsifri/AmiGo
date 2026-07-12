from typing import NamedTuple

import numpy as np
import scipy.sparse
import trimesh as tm
from trimesh.geometry import index_sparse

from src.consts import EPSILON


class CurvatureResult(NamedTuple):
    """
    Attributes:
        dminf (ndarray (nf, 3)): min-curvature principal directions
        dmaxf (ndarray (nf, 3)): max-curvature principal directions
        kminf (ndarray (nf,)): min principal curvatures
        kmaxf (ndarray (nf,)): max principal curvatures
    """
    dminf: np.ndarray
    dmaxf: np.ndarray
    kminf: np.ndarray
    kmaxf: np.ndarray


def vertex_normals(mesh: tm.Trimesh) -> np.ndarray:
    """Per-vertex normals as area-weighted averages of incident face normals.

    Args:
        mesh: Input triangular mesh.

    Returns:
        Array of shape (nv, 3) with unit-length per-vertex normals.
    """
    nv = len(mesh.vertices)
    weighted_normals = mesh.face_normals * mesh.area_faces[:, None]  # (nf, 3)

    # (nv × nf) adjacency matrix: entry [v, f] = 1 whenever vertex v belongs to face f.
    face_vertex_mat = index_sparse(nv, mesh.faces)

    # Multiplying by weighted_normals accumulates each face's contribution into its vertices.
    normals = np.asarray(face_vertex_mat @ weighted_normals)  # (nv, 3)

    scale = np.linalg.norm(normals, axis=1, keepdims=True)
    scale = np.where(scale < EPSILON, 1.0, scale)
    return normals / scale


def _edge_vectors(mesh: tm.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-face edge vectors, each opposite the vertex with the same index.

    Args:
        mesh: Input triangular mesh.

    Returns:
        Tuple (e0, e1, e2), each of shape (nf, 3). e_i is the edge opposite vertex i.
    """
    e0 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]]
    e1 = mesh.vertices[mesh.faces[:, 0]] - mesh.vertices[mesh.faces[:, 2]]
    e2 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]
    return e0, e1, e2


def shape_operator_ftf(mesh: tm.Trimesh) -> CurvatureResult:
    """Estimate per-face principal curvatures via the FTF (gradient-of-vertex-normals) method.

    Based on Rusinkiewicz 2004, "Estimating Curvatures and Their Derivatives on Triangle Meshes".

    Args:
        mesh: Input triangular mesh.

    Returns:
        CurvatureResult with fields dminf (nf, 3), dmaxf (nf, 3), kminf (nf,),
        kmaxf (nf,) — principal directions and curvatures per face, with kminf ≤ kmaxf.
    """
    nf = len(mesh.faces)
    face_normals = mesh.face_normals  # (nf, 3)

    e0, e1, e2 = _edge_vectors(mesh)

    safe_areas = np.maximum(mesh.area_faces, EPSILON)
    inv_face_areas = (1.0 / safe_areas)[:, None]  # (nf, 1)

    # cross(face_normals, e) rotates e by 90° within the tangent plane,
    # scaling by 0.5/area gives the FTF gradient weight.
    edge_gradients = [0.5 * inv_face_areas * np.cross(face_normals, e) for e in (e0, e1, e2)]  # each (nf, 3)

    vertex_norms = vertex_normals(mesh)

    # shape_mat[f] = Σ_i  outer(vertex_norms[faces[f,i]], edge_gradients[i][f])
    # rank-1 outer product per vertex, summed to form the 3×3 per-face shape operator.
    # einsum 'fi,fj->fij' computes all nf outer products in a single batched call.
    shape_mat = sum(
        np.einsum('fi,fj->fij', vertex_norms[mesh.faces[:, k]], edge_gradients[k]) for k in range(3)
    )  # (nf, 3, 3)

    # Project into the tangent plane: tangent_proj[f] = I − face_normals[f]⊗face_normals[f]
    # zeroes the normal component. np.eye(3) broadcasts from (3,3) to (nf,3,3).
    tangent_proj = np.eye(3) - np.einsum('fi,fj->fij', face_normals, face_normals)  # (nf, 3, 3)

    shape_mat = tangent_proj @ shape_mat  # batched (nf,3,3) @ (nf,3,3) — @ broadcasts over axis 0

    # Symmetrize: the shape operator is self-adjoint; small asymmetry is numerical noise.
    shape_mat = (shape_mat + shape_mat.transpose(0, 2, 1)) / 2  # (nf, 3, 3)

    # eigh exploits symmetry (faster + more stable than eig) and guarantees real
    # eigenvalues in ascending algebraic order — no np.real() needed afterward.
    eigenvalues, eigenvectors = np.linalg.eigh(shape_mat)  # (nf, 3), (nf, 3, 3) columns = eigenvectors

    # Sort per-face eigenvalues by absolute value to classify degenerate configurations.
    abs_eigenvalues = np.abs(eigenvalues)  # (nf, 3)
    abs_sort_idx = np.argsort(abs_eigenvalues, axis=1)  # (nf, 3)
    sorted_abs_eigenvalues = np.take_along_axis(abs_eigenvalues, abs_sort_idx, axis=1)  # (nf, 3) ascending |d|

    # Three mutually exhaustive cases:
    # mask_flat:        all |d| ≈ 0 → fully flat region
    # mask_one_curved:  exactly one |d| nonzero → one principal curvature, one flat direction
    # mask_generic:     everything else (normal case or one near-zero from the tangent projection)
    mask_flat = np.all(sorted_abs_eigenvalues < EPSILON, axis=1)
    # No need to check [:, 0] < EPSILON: ascending sort guarantees [:, 0] ≤ [:, 1],
    # so [:, 1] < EPSILON implies [:, 0] < EPSILON.
    mask_one_curved = (sorted_abs_eigenvalues[:, 1] < EPSILON) & (sorted_abs_eigenvalues[:, 2] >= EPSILON)
    mask_generic = ~mask_flat & ~mask_one_curved

    dminf = np.zeros((nf, 3))
    dmaxf = np.zeros((nf, 3))
    kminf = np.zeros(nf)
    kmaxf = np.zeros(nf)

    # ------------------------------------------------------------------ #
    # Case 1 — generic, or one zero-curvature direction
    # ------------------------------------------------------------------ #
    if np.any(mask_generic):
        # After tangent projection one of the three eigenvectors lies approximately
        # along the face normal and must be discarded. We identify it as the eigenvector
        # whose projection onto the in-plane edge span {e0, e1} has the smallest norm.
        eigvecs = eigenvectors[mask_generic]  # (n_generic, 3, 3)
        eigenvals = eigenvalues[mask_generic]  # (n_generic, 3)
        n_generic = int(mask_generic.sum())

        # Stack edge vectors as a 3×2 basis per face, then project each eigenvector (row of eigvecsᵀ) onto this basis.
        edge_span = np.stack([e0[mask_generic], e1[mask_generic]], axis=-1)  # (n_generic, 3, 2)

        # eigvecs.transpose(0,2,1) turns eigenvectors into rows → (n_generic, 3, 3)
        # batched @ edge_span (n_generic, 3, 2) → (n_generic, 3, 2): row i is the 2-D projection of eigenvector i
        normal_proj_norms = np.linalg.norm(eigvecs.transpose(0, 2, 1) @ edge_span, axis=-1)  # (n_generic, 3)

        # Smallest norm → most "normal-like" eigenvector.
        normal_col = np.argmin(normal_proj_norms, axis=1)  # (n_generic,) column index of the normal eigenvector

        # Look up which two column indices to keep after removing normal_col.
        # For example, if we eliminate index 0 - we need to keep indices [1, 2] etc...
        kept_col_pairs = np.array([[1, 2], [0, 2], [0, 1]])
        kept_cols = kept_col_pairs[normal_col]  # (n_generic, 2)

        # Gather the two kept eigenvectors per face with 3-D advanced indexing:
        # tangent_eigvecs[f, j, k] = eigvecs[f, j, kept_cols[f, k]]
        face_sel = np.arange(n_generic)[:, np.newaxis, np.newaxis]  # (n_generic, 1, 1)
        row_sel = np.arange(3)[np.newaxis, :, np.newaxis]  # (1, 3, 1)
        col_sel = kept_cols[:, np.newaxis, :]  # (n_generic, 1, 2)
        tangent_eigvecs = eigvecs[face_sel, row_sel, col_sel]  # (n_generic, 3, 2)
        tangent_eigenvals = eigenvals[np.arange(n_generic)[:, None], kept_cols]  # (n_generic, 2)

        # kept_col_pairs always selects indices in ascending order ([1,2], [0,2], [0,1]),
        # and eigh returns eigenvalues in ascending algebraic order, so tangent_eigenvals
        # is already sorted — no explicit sort needed here.
        dminf[mask_generic] = tangent_eigvecs[:, :, 0]
        dmaxf[mask_generic] = tangent_eigvecs[:, :, 1]
        kminf[mask_generic] = tangent_eigenvals[:, 0]
        kmaxf[mask_generic] = tangent_eigenvals[:, 1]

    # ------------------------------------------------------------------ #
    # Case 2 — fully flat (all |d| ≈ 0)
    # ------------------------------------------------------------------ #
    # Eigenvectors are numerically arbitrary in a near-zero matrix; build an
    # orthonormal tangent frame from e0 directly instead.
    if np.any(mask_flat):
        tangent_dir1 = e0[mask_flat] / np.linalg.norm(e0[mask_flat], axis=1, keepdims=True)
        tangent_dir2 = np.cross(tangent_dir1, face_normals[mask_flat])
        tangent_dir2 /= np.linalg.norm(tangent_dir2, axis=1, keepdims=True)
        dminf[mask_flat] = tangent_dir1
        dmaxf[mask_flat] = tangent_dir2
        # kminf, kmaxf stay 0

    # ------------------------------------------------------------------ #
    # Case 3 — one nonzero curvature, one flat direction
    # ------------------------------------------------------------------ #
    # The tangent projection annihilates the face normal, so any eigenvector
    # with eigenvalue ≠ 0 must already lie in the tangent plane — no projection needed.
    # The second tangent direction is recovered via the cross product with face_normals.
    if np.any(mask_one_curved):
        n_one_curved = int(mask_one_curved.sum())
        eigvecs = eigenvectors[mask_one_curved]  # (n_one_curved, 3, 3)
        eigenvals = eigenvalues[mask_one_curved]  # (n_one_curved, 3)
        abs_sort_idx_subset = abs_sort_idx[mask_one_curved]  # (n_one_curved, 3) column indices sorted by |d|

        # abs_sort_idx_subset[:, 2] is the column of the eigenvector with the largest |d|.
        # Advanced indexing: principal_dir[f, j] = eigvecs[f, j, dominant_col[f]]
        dominant_col = abs_sort_idx_subset[:, 2]  # (n_one_curved,)
        face_sel = np.arange(n_one_curved)[:, np.newaxis]  # (n_one_curved, 1)
        row_sel = np.arange(3)[np.newaxis, :]  # (1, 3)
        principal_dir = eigvecs[face_sel, row_sel, dominant_col[:, np.newaxis]]  # (n_one_curved, 3)
        principal_curv = eigenvals[np.arange(n_one_curved), dominant_col]  # (n_one_curved,)

        flat_dir = np.cross(principal_dir, face_normals[mask_one_curved])
        flat_dir /= np.linalg.norm(flat_dir, axis=1, keepdims=True)

        dminf[mask_one_curved] = principal_dir
        dmaxf[mask_one_curved] = flat_dir
        kminf[mask_one_curved] = principal_curv
        # kmaxf stays 0

    # Enforce kminf ≤ kmaxf globally with a single swap pass.
    # Case 1 never needs it (shown above); case 2 never needs it (both curvatures are 0).
    # Only case 3 faces where principal_curv > 0 will be swapped.
    needs_swap = kminf > kmaxf
    dminf[needs_swap], dmaxf[needs_swap] = dmaxf[needs_swap].copy(), dminf[needs_swap].copy()
    kminf[needs_swap], kmaxf[needs_swap] = kmaxf[needs_swap].copy(), kminf[needs_swap].copy()

    return CurvatureResult(dminf, dmaxf, kminf, kmaxf)


def edge_basis(mesh: tm.Trimesh, e0: np.ndarray | None = None) -> scipy.sparse.csr_matrix:
    """Build the 2nf × 3nf matrix that projects a per-face 3-D vector field into
    a per-face 2-D orthonormal tangent basis aligned with e0.

    Args:
        mesh: Input triangular mesh.
        e0: Optional precomputed edge vectors of shape (nf, 3). Computed from mesh
            if not provided.

    Returns:
        Sparse matrix of shape (2·nf, 3·nf).
    """
    if e0 is None:
        e0 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]]

    basis_dir1 = e0 / np.linalg.norm(e0, axis=1, keepdims=True)  # first basis direction
    basis_dir2 = np.cross(mesh.face_normals, basis_dir1, axis=1)  # second direction, orthogonal to both

    nf = len(mesh.faces)

    # Two sparse (nf × 3nf) "projection row" matrices, one per basis vector.
    # Columns are grouped as [x-coords of all faces | y-coords | z-coords],
    # matching the Fortran-order flattening convention used throughout shapeop().
    row_indices = np.tile(np.arange(nf), 3)
    col_indices = np.concatenate([np.arange(nf), np.arange(nf) + nf, np.arange(nf) + 2 * nf])

    proj_mat_dir1 = scipy.sparse.csr_matrix(
        (basis_dir1.flatten(order='F'), (row_indices, col_indices)), shape=(nf, 3 * nf)
    )
    proj_mat_dir2 = scipy.sparse.csr_matrix(
        (basis_dir2.flatten(order='F'), (row_indices, col_indices)), shape=(nf, 3 * nf)
    )

    return scipy.sparse.vstack([proj_mat_dir1, proj_mat_dir2])  # (2nf × 3nf)


def shapeop(mesh: tm.Trimesh) -> scipy.sparse.csr_matrix:
    """Return the 2nf × 2nf shape operator matrix SO = Vᵀ D V.

    V encodes curvature directions in the per-face tangent basis; D holds the
    principal curvatures on its diagonal. Each 2×2 block corresponds to one face.

    Args:
        mesh: Input triangular mesh.

    Returns:
        Sparse matrix of shape (2·nf, 2·nf).
    """
    nf = len(mesh.faces)

    # Compute e0 once and pass into edge_basis to avoid a redundant subtraction.
    e0 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]]

    curvature = shape_operator_ftf(mesh)
    edge_basis_mat = edge_basis(mesh, e0)

    # Project the 3-D principal directions into the 2-D per-face tangent basis.
    dmin_in_basis = (edge_basis_mat @ curvature.dminf.flatten(order='F')).reshape(-1, 2, order='F')  # (nf, 2)
    dmax_in_basis = (edge_basis_mat @ curvature.dmaxf.flatten(order='F')).reshape(-1, 2, order='F')  # (nf, 2)

    dmin_basis1, dmin_basis2 = dmin_in_basis[:, 0], dmin_in_basis[:, 1]
    dmax_basis1, dmax_basis2 = dmax_in_basis[:, 0], dmax_in_basis[:, 1]

    # dir_matrix is 2nf × 2nf: its 2×2 blocks hold the curvature directions
    # expressed in the local tangent basis.
    dir_matrix = scipy.sparse.bmat([
        [scipy.sparse.diags(dmin_basis1), scipy.sparse.diags(dmin_basis2)],
        [scipy.sparse.diags(dmax_basis1), scipy.sparse.diags(dmax_basis2)],
    ])

    # curv_matrix is block-diagonal with the two principal curvatures per face.
    curv_matrix = scipy.sparse.bmat([
        [scipy.sparse.diags(curvature.kminf), scipy.sparse.csr_matrix((nf, nf))],
        [scipy.sparse.csr_matrix((nf, nf)), scipy.sparse.diags(curvature.kmaxf)],
    ])

    return dir_matrix.T @ curv_matrix @ dir_matrix
