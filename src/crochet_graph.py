from typing import NamedTuple

import numpy as np
import trimesh as tm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.consts import EPSILON


class CrochetGraph(NamedTuple):
    """Crochet graph built from a parameterized mesh.

    Attributes:
        vertices ((n, 3), float): all sampled stitch locations in 3D, concatenated across rows
        row_edges ((m, 2), int): index pairs of adjacent stitches within each row;
            the last pair in each row wraps around to close the loop
        column_edges ((p, 2), int): DTW-aligned index pairs between consecutive rows,
            encoding increases and decreases where row lengths differ
        connectivity (list of (k, 2) int): per-row-pair DTW paths using local row indices
    """
    vertices: np.ndarray
    row_edges: np.ndarray
    column_edges: np.ndarray
    connectivity: list


def get_crochet_graph(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                      stitch_size: float) -> CrochetGraph:
    """Build the full crochet graph: stitch vertices, row edges, and column edges.

    Args:
        mesh: the cut mesh returned by ``compute_row_column_order``
        row_order ((v,), float): geodesic-distance field (u-axis) on the cut mesh vertices
        column_order ((v,), float): tangent scalar field (v-axis) on the cut mesh vertices
        stitch_size (float): uniform sampling spacing in UV space; controls stitch density

    Returns:
        A ``CrochetGraph`` with:
            - vertices ((n, 3), float): all sampled stitch locations in 3D, concatenated across rows
            - row_edges ((m, 2), int): index pairs of adjacent stitches within each row;
              the last pair in each row wraps around to close the loop
            - column_edges ((p, 2), int): DTW-aligned index pairs between consecutive rows,
              encoding increases and decreases where row lengths differ
            - connectivity (list of (k, 2) int): per-row-pair DTW paths using local row indices
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

    column_edge_list = []
    connectivity = []
    for (r1, o1), (r2, o2) in zip(zip(row_separated_vertices[:-1], row_offsets[:-1]),
                                  zip(row_separated_vertices[1:], row_offsets[1:])):
        _, path = fastdtw(r1, r2, dist=euclidean)
        path = np.array(path)
        column_edge_list.append(np.stack([path[:, 0] + o1, path[:, 1] + o2], axis=1))
        connectivity.append(np.array(path))
    column_edges = np.concatenate(column_edge_list, axis=0)

    return CrochetGraph(vertices, row_edges, column_edges, connectivity)


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
