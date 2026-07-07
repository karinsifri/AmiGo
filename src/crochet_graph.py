from typing import NamedTuple

import gpytoolbox as gpy
import numpy as np
import trimesh as tm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.consts import CREASE_ALIGNMENT_THRESHOLD, CREASE_CURVATURE_THRESHOLD, EPSILON
from src.shapeop import shape_operator_ftf


class CrochetGraph(NamedTuple):
    """Crochet graph built from a parameterized mesh.

    Attributes:
        vertices ((n, 3), float): all sampled stitch locations in 3D, concatenated across rows
        row_edges ((m, 2), int): index pairs of adjacent stitches within each row;
            the last pair in each row wraps around to close the loop
        column_edges ((p, 2), int): DTW-aligned index pairs between consecutive rows,
            encoding increases and decreases where row lengths differ
        connectivity (list of (k, 2) int): per-row-pair DTW paths using local row indices
        creases ((n,), int8): crease label per stitch vertex — 1 for BLO, -1 for FLO, 0 for regular
        split_creases (list of (k,) int8): creases split by row; entry i holds labels for row i;
            the last row is omitted because its vertices are never stitch roots
    """
    vertices: np.ndarray
    row_edges: np.ndarray
    column_edges: np.ndarray
    connectivity: list
    creases: np.ndarray
    split_creases: list[np.ndarray]


def get_crochet_graph(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                      stitch_size: float, use_creases: bool) -> CrochetGraph:
    """Build the full crochet graph: stitch vertices, row edges, and column edges.

    Args:
        mesh: the cut mesh returned by ``compute_row_column_order``
        row_order ((v,), float): geodesic-distance field (u-axis) on the cut mesh vertices
        column_order ((v,), float): tangent scalar field (v-axis) on the cut mesh vertices
        stitch_size (float): uniform sampling spacing in UV space; controls stitch density
        use_creases (bool): if True, classify stitch vertices as BLO/FLO/regular using principal curvatures;
            if False, all creases are set to 0

    Returns:
        A ``CrochetGraph`` with:
            - vertices ((n, 3), float): all sampled stitch locations in 3D, concatenated across rows
            - row_edges ((m, 2), int): index pairs of adjacent stitches within each row;
              the last pair in each row wraps around to close the loop
            - column_edges ((p, 2), int): DTW-aligned index pairs between consecutive rows,
              encoding increases and decreases where row lengths differ
            - connectivity (list of (k, 2) int): per-row-pair DTW paths using local row indices
            - creases ((n,), int8): crease label per stitch vertex — 1 for BLO, -1 for FLO, 0 for regular
            - split_creases (list of (k,) int8): creases split by row, one entry per row except the last
    """
    row_separated_vertices = calculate_crochet_graph_vertices(mesh, row_order, column_order, stitch_size)

    vertices = np.concatenate(row_separated_vertices)

    row_lengths = [len(r) for r in row_separated_vertices]
    row_offsets = np.concatenate([[0], np.cumsum(row_lengths[:-1])])

    row_edge_list = []
    for offset, row_len in zip(row_offsets, row_lengths):
        local = np.arange(row_len)
        row_edge_list.append(np.stack([local + offset, np.roll(local, -1) + offset], axis=1))
    row_edges = np.concatenate(row_edge_list, axis=0)

    creases = np.zeros(len(vertices), dtype=np.int8)
    if use_creases:
        creases = get_crease_vertices(mesh, vertices, row_order, row_edges)

    column_edge_list = []
    connectivity = []
    split_creases = []
    for (r1, o1), (r2, o2) in zip(zip(row_separated_vertices[:-1], row_offsets[:-1]),
                                  zip(row_separated_vertices[1:], row_offsets[1:])):
        _, path = fastdtw(r1, r2, dist=euclidean)
        path = np.array(path)
        column_edge_list.append(np.stack([path[:, 0] + o1, path[:, 1] + o2], axis=1))
        connectivity.append(np.array(path))
        split_creases.append(creases[o1:o2].copy())
    column_edges = np.concatenate(column_edge_list, axis=0)

    return CrochetGraph(vertices, row_edges, column_edges, connectivity, creases, split_creases)


def calculate_crochet_graph_vertices(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                                     stitch_size: float) -> list[np.ndarray]:
    """Sample stitch locations on the mesh by projecting a uniform UV grid back to 3D.

    Uses ``row_order`` and ``column_order`` as a UV parameterization.  A regular grid with
    spacing ``stitch_size`` is overlaid on the UV domain; grid points that fall inside the
    mesh's UV footprint are kept and lifted back to 3D via barycentric interpolation.  The
    resulting 3D points are grouped by their u-axis grid index, yielding one array per row.

    An extra point at ``(row_order.max(), 0)`` is appended before filtering so that the seam
    vertex (where column_order wraps to 0) is always represented in the first column slot of
    the last row.

    Args:
        mesh: the cut mesh returned by ``compute_row_column_order``
        row_order ((v,), float): geodesic-distance field (u-axis) on the cut mesh vertices
        column_order ((v,), float): tangent scalar field (v-axis) on the cut mesh vertices
        stitch_size (float): grid spacing in UV space; smaller values produce denser stitches

    Returns:
        A list of length ``n_rows`` where ``result[i]`` is an ``(k_i, 3)`` float array of
        3D stitch positions belonging to the i-th row (u-index i).
    """
    grid_u, grid_v = np.meshgrid(np.arange(0, row_order.max(), stitch_size),
                                 np.arange(0, column_order.max(), stitch_size))
    u_grid_idx = np.tile(np.arange(grid_u.shape[1]), grid_u.shape[0])

    flatten_mesh = tm.Trimesh(np.vstack([row_order, column_order, np.zeros_like(row_order)]).T, mesh.faces)

    sampled_points = np.stack((np.append(grid_u.ravel(), row_order.max()),
                               np.append(grid_v.ravel(), 0),
                               np.zeros((grid_u.size + 1,))), axis=-1)

    _, point_dist, face_id = tm.proximity.closest_point(flatten_mesh, sampled_points)
    on_mesh = point_dist < EPSILON
    sampled_points = sampled_points[on_mesh]
    face_id = face_id[on_mesh]

    sampled_barycentric = tm.triangles.points_to_barycentric(flatten_mesh.vertices[mesh.faces[face_id]],
                                                             sampled_points)

    sampled_3d = tm.triangles.barycentric_to_points(mesh.vertices[mesh.faces[face_id]], sampled_barycentric)

    row_idx = np.append(u_grid_idx, u_grid_idx.max() + 1)[on_mesh]

    row_separated = [sampled_3d[row_idx == i] for i in np.arange(row_idx.max() + 1)]

    return row_separated


def get_crease_vertices(mesh: tm.Trimesh, crochet_graph_vertices: np.ndarray, row_order: np.ndarray,
                        row_graph_edges: np.ndarray) -> np.ndarray:
    """Classify each stitch vertex as BLO, FLO, or regular based on principal curvature.

    A vertex is BLO (back loop only) when the mesh curves sharply in the column direction with
    positive (convex) curvature: the max-curvature direction is orthogonal to the isoline and
    ``kmaxf > CREASE_CURVATURE_THRESHOLD``.  FLO (front loop only) is the analogous condition
    for the min-curvature direction with negative (concave) curvature.  A crease is only
    assigned when two consecutive row-neighbours both qualify, to suppress isolated detections.

    Args:
        mesh: the cut mesh returned by ``compute_row_column_order``
        crochet_graph_vertices ((n, 3), float): stitch vertex positions in 3D
        row_order ((v,), float): geodesic-distance field (u-axis) on the cut mesh vertices;
            its gradient defines the across-row direction, whose tangent-plane rotation gives
            the isoline direction
        row_graph_edges ((m, 2), int): index pairs of adjacent stitches within each row

    Returns:
        ((n,), int8): crease label per stitch vertex — 1 for BLO, -1 for FLO, 0 for regular
    """
    _, _, appropriate_faces = tm.proximity.closest_point(mesh, crochet_graph_vertices)

    dminf, dmaxf, kminf, kmaxf = shape_operator_ftf(mesh)

    gradients = np.reshape(gpy.grad(mesh.vertices, mesh.faces) @ row_order, (-1, 3), order='F')
    rotated = np.cross(mesh.face_normals, gradients, axis=1)
    isoline_direction = rotated / np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), EPSILON)

    flo = (np.abs(np.linalg.vecdot(dminf[appropriate_faces], isoline_direction[appropriate_faces], axis=1)) <
           CREASE_ALIGNMENT_THRESHOLD) & (kminf[appropriate_faces] < -CREASE_CURVATURE_THRESHOLD)
    blo = (np.abs(np.linalg.vecdot(dmaxf[appropriate_faces], isoline_direction[appropriate_faces], axis=1)) <
           CREASE_ALIGNMENT_THRESHOLD) & (kmaxf[appropriate_faces] > CREASE_CURVATURE_THRESHOLD)

    # check for two consecutive vertices in a row
    creases = np.zeros(len(crochet_graph_vertices), dtype=np.int8)
    creases[np.unique(row_graph_edges[np.all(blo[row_graph_edges], axis=1)])] = 1
    creases[np.unique(row_graph_edges[np.all(flo[row_graph_edges], axis=1)])] = -1

    return creases
