import numpy as np
import scipy.sparse
import trimesh as tm
from typing import NamedTuple

from src.consts import EPSILON

# Matches MATLAB's 1e-15 threshold for eigenvalue near-zero detection.
# Kept separate from EPSILON because geometric tolerances (vertex distances,
# normal magnitudes) are coarser than floating-point near-zero in a 3×3 matrix.
_EIG_EPS = 1e-15


class CurvatureResult(NamedTuple):
    dminf: np.ndarray   # (nf, 3)  min-curvature principal directions
    dmaxf: np.ndarray   # (nf, 3)  max-curvature principal directions
    kminf: np.ndarray   # (nf,)    min principal curvatures
    kmaxf: np.ndarray   # (nf,)    max principal curvatures


def vertex_normals(mesh: tm.Trimesh) -> np.ndarray:
    """Per-vertex normals as area-weighted averages of incident face normals.

    Args:
        mesh: Input triangular mesh.

    Returns:
        Array of shape (nv, 3) with unit-length per-vertex normals.
    """
    nv, nf = len(mesh.vertices), len(mesh.faces)
    weighted = mesh.face_normals * mesh.area_faces[:, None]  # (nf, 3)

    # Build a (nv × nf) incidence matrix A where A[v, f] = 1 whenever vertex v
    # belongs to face f. Then A @ weighted accumulates each face's weighted normal
    # into its three vertices — equivalent to MATLAB's sparse(I, J, S, nv, 3).
    # Faster than np.add.at because the sparse matmul runs in compiled C.
    rows = mesh.faces.flatten()           # (3·nf,) vertex index for each (face, corner)
    cols = np.repeat(np.arange(nf), 3)   # (3·nf,) face index for each (face, corner)
    A = scipy.sparse.csr_matrix((np.ones(3 * nf), (rows, cols)), shape=(nv, nf))
    vertex_norm = np.asarray(A @ weighted)  # (nv, 3)

    scale = np.linalg.norm(vertex_norm, axis=1, keepdims=True)
    scale = np.where(scale < EPSILON, 1.0, scale)
    return vertex_norm / scale


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
    n = mesh.face_normals  # (nf, 3)

    e0, e1, e2 = _edge_vectors(mesh)

    # cross(n, e) rotates e by 90° within the tangent plane, equivalent to
    # MATLAB's mesh.rotate_vf(). Scaling by 0.5/area gives the FTF gradient weight.
    ita = (1.0 / mesh.area_faces)[:, None]  # (nf, 1) inverse face areas
    ge = [0.5 * ita * np.cross(n, e) for e in (e0, e1, e2)]  # each (nf, 3)

    vn = vertex_normals(mesh)

    # s[f] = Σ_i  outer(vn[faces[f,i]], ge[i][f]) — rank-1 outer product per vertex,
    # summed to form the 3×3 per-face shape operator matrix.
    # einsum 'fi,fj->fij' computes all nf outer products in a single batched call.
    s = sum(
        np.einsum('fi,fj->fij', vn[mesh.faces[:, k]], ge[k]) for k in range(3)
    )  # (nf, 3, 3)

    # Project into the tangent plane: P[f] = I − n[f]⊗n[f] zeroes the normal component.
    # np.eye(3) broadcasts from (3,3) to (nf,3,3) via NumPy's standard broadcasting.
    P = np.eye(3) - np.einsum('fi,fj->fij', n, n)  # (nf, 3, 3)
    s = P @ s  # batched (nf,3,3) @ (nf,3,3) — NumPy's @ operator broadcasts over axis 0

    # Symmetrize: the shape operator is self-adjoint; small asymmetry is numerical noise.
    s = (s + s.transpose(0, 2, 1)) / 2  # (nf, 3, 3)

    # eigh exploits symmetry (faster + more stable than eig) and guarantees real
    # eigenvalues in ascending algebraic order — no np.real() needed afterward.
    d, v = np.linalg.eigh(s)  # d: (nf, 3) eigenvalues, v: (nf, 3, 3) eigenvectors as columns

    # Sort per-face eigenvalues by absolute value to classify degenerate configurations.
    abs_d = np.abs(d)                                                  # (nf, 3)
    sort_idx = np.argsort(abs_d, axis=1)                               # (nf, 3)
    abs_d_s = np.take_along_axis(abs_d, sort_idx, axis=1)             # (nf, 3) ascending |d|

    # Three mutually exhaustive cases (mirrors the original per-face if/elif chain):
    # mask2: all |d| ≈ 0 → fully flat region
    # mask3: exactly one |d| nonzero → one principal curvature, one flat direction
    # mask1: everything else (normal case or one near-zero from the tangent projection)
    mask2 = np.all(abs_d_s < _EIG_EPS, axis=1)
    mask3 = (abs_d_s[:, 1] < _EIG_EPS) & (abs_d_s[:, 2] >= _EIG_EPS)
    mask1 = ~mask2 & ~mask3

    dminf = np.zeros((nf, 3))
    dmaxf = np.zeros((nf, 3))
    kminf = np.zeros(nf)
    kmaxf = np.zeros(nf)

    # ------------------------------------------------------------------ #
    # Case 1 — generic, or one zero-curvature direction
    # ------------------------------------------------------------------ #
    # After tangent projection one of the three eigenvectors lies approximately
    # along the face normal and must be discarded. We identify it as the eigenvector
    # whose projection onto the in-plane edge span {e0, e1} has the smallest norm.
    if np.any(mask1):
        v1 = v[mask1]    # (n1, 3, 3)
        d1 = d[mask1]    # (n1, 3)
        n1 = int(mask1.sum())

        # Stack edge vectors as a 3×2 basis per face, then project each eigenvector
        # (row of v1ᵀ) onto this basis. Smallest norm → most "normal-like" eigenvector.
        e_span = np.stack([e0[mask1], e1[mask1]], axis=-1)              # (n1, 3, 2)
        # v1.transpose(0,2,1) turns eigenvectors into rows → (n1, 3, 3)
        # batched @ e_span (n1, 3, 2) → (n1, 3, 2): row i is the 2-D projection of eigenvector i
        proj_norms = np.linalg.norm(v1.transpose(0, 2, 1) @ e_span, axis=-1)  # (n1, 3)
        rm_idx = np.argmin(proj_norms, axis=1)                          # (n1,)

        # Look up which two column indices to keep after removing rm_idx.
        remaining = np.array([[1, 2], [0, 2], [0, 1]])
        pair_idx = remaining[rm_idx]                                    # (n1, 2)

        # Gather the two kept eigenvectors per face with 3-D advanced indexing:
        # v_pair[f, j, k] = v1[f, j, pair_idx[f, k]]
        fi = np.arange(n1)[:, np.newaxis, np.newaxis]                  # (n1, 1, 1)
        ji = np.arange(3)[np.newaxis, :, np.newaxis]                   # (1, 3, 1)
        ki = pair_idx[:, np.newaxis, :]                                 # (n1, 1, 2)
        v_pair = v1[fi, ji, ki]                                         # (n1, 3, 2)
        d_pair = d1[np.arange(n1)[:, None], pair_idx]                  # (n1, 2)

        # Sort by algebraic value so kminf ≤ kmaxf.
        s_i = np.argsort(d_pair, axis=1)                               # (n1, 2)
        s_i3 = np.broadcast_to(s_i[:, np.newaxis, :], (n1, 3, 2))     # broadcast to match v_pair shape
        v_s = np.take_along_axis(v_pair, s_i3, axis=2)                 # (n1, 3, 2)
        d_s = np.take_along_axis(d_pair, s_i, axis=1)                  # (n1, 2)

        dminf[mask1] = v_s[:, :, 0]
        dmaxf[mask1] = v_s[:, :, 1]
        kminf[mask1] = d_s[:, 0]
        kmaxf[mask1] = d_s[:, 1]

    # ------------------------------------------------------------------ #
    # Case 2 — fully flat (all |d| ≈ 0)
    # ------------------------------------------------------------------ #
    # Eigenvectors are numerically arbitrary in a near-zero matrix; build an
    # orthonormal tangent frame from e0 directly instead.
    if np.any(mask2):
        v1 = e0[mask2] / np.linalg.norm(e0[mask2], axis=1, keepdims=True)
        v2 = np.cross(v1, n[mask2])
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
        dminf[mask2] = v1
        dmaxf[mask2] = v2
        # kminf, kmaxf stay 0

    # ------------------------------------------------------------------ #
    # Case 3 — one nonzero curvature, one flat direction
    # ------------------------------------------------------------------ #
    # The tangent projection P annihilates the face normal, so any eigenvector
    # with d ≠ 0 must already lie in the tangent plane — no projection needed.
    # The second tangent direction is recovered via the cross product with n.
    if np.any(mask3):
        n3 = int(mask3.sum())
        v3 = v[mask3]          # (n3, 3, 3)
        d3 = d[mask3]          # (n3, 3)
        si3 = sort_idx[mask3]  # (n3, 3) column indices sorted by |d|

        # si3[:, 2] is the column index of the eigenvector with the largest |d|.
        # Advanced indexing: vp[f, j] = v3[f, j, si3[f, 2]]
        large_i = si3[:, 2]                                             # (n3,)
        fi3 = np.arange(n3)[:, np.newaxis]                             # (n3, 1)
        ji3 = np.arange(3)[np.newaxis, :]                              # (1, 3)
        vp = v3[fi3, ji3, large_i[:, np.newaxis]]                      # (n3, 3) principal direction
        dp = d3[np.arange(n3), large_i]                                # (n3,) its curvature

        vq = np.cross(vp, n[mask3])
        vq /= np.linalg.norm(vq, axis=1, keepdims=True)

        d_pair = np.stack([dp, np.zeros(n3)], axis=1)                  # (n3, 2)
        v_pair = np.stack([vp, vq], axis=2)                            # (n3, 3, 2)

        # Sort so kminf ≤ kmaxf (mirrors original: sort([dd1, 0])).
        s_i = np.argsort(d_pair, axis=1)                               # (n3, 2)
        s_i3 = np.broadcast_to(s_i[:, np.newaxis, :], (n3, 3, 2))
        v_s = np.take_along_axis(v_pair, s_i3, axis=2)
        d_s = np.take_along_axis(d_pair, s_i, axis=1)

        dminf[mask3] = v_s[:, :, 0]
        dmaxf[mask3] = v_s[:, :, 1]
        kminf[mask3] = d_s[:, 0]
        kmaxf[mask3] = d_s[:, 1]

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

    ne1 = e0 / np.linalg.norm(e0, axis=1, keepdims=True)   # first basis direction
    ne2 = np.cross(mesh.face_normals, ne1, axis=1)          # second direction, orthogonal to both

    nf = len(mesh.faces)

    # Two sparse (nf × 3nf) "projection row" matrices, one per basis vector.
    # Columns are grouped as [x-coords of all faces | y-coords | z-coords],
    # matching the Fortran-order flattening convention used throughout shapeop().
    i = np.tile(np.arange(nf), 3)
    j = np.concatenate([np.arange(nf), np.arange(nf) + nf, np.arange(nf) + 2 * nf])

    b1 = scipy.sparse.csr_matrix((ne1.flatten(order='F'), (i, j)), shape=(nf, 3 * nf))
    b2 = scipy.sparse.csr_matrix((ne2.flatten(order='F'), (i, j)), shape=(nf, 3 * nf))

    return scipy.sparse.vstack([b1, b2])  # (2nf × 3nf)


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

    result = shape_operator_ftf(mesh)
    eb = edge_basis(mesh, e0)

    # Project the 3-D principal directions into the 2-D per-face tangent basis.
    # Fortran-order flattening matches the column-major layout assumed by edge_basis.
    dminfb = (eb @ result.dminf.flatten(order='F')).reshape(-1, 2, order='F')  # (nf, 2)
    dmaxfb = (eb @ result.dmaxf.flatten(order='F')).reshape(-1, 2, order='F')  # (nf, 2)

    v11, v12 = dminfb[:, 0], dminfb[:, 1]
    v21, v22 = dmaxfb[:, 0], dmaxfb[:, 1]

    # V is a 2nf × 2nf block-diagonal matrix whose 2×2 blocks hold the curvature
    # directions expressed in the local tangent basis.
    v = scipy.sparse.bmat([
        [scipy.sparse.diags(v11), scipy.sparse.diags(v12)],
        [scipy.sparse.diags(v21), scipy.sparse.diags(v22)],
    ])

    # D is block-diagonal with the two principal curvatures per face.
    d = scipy.sparse.bmat([
        [scipy.sparse.diags(result.kminf), scipy.sparse.csr_matrix((nf, nf))],
        [scipy.sparse.csr_matrix((nf, nf)), scipy.sparse.diags(result.kmaxf)],
    ])

    return v.T @ d @ v
