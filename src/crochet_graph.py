import numpy as np
import trimesh as tm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from consts import EPSILON


def get_row_connectivity(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                         stitch_size: float) -> list[np.ndarray]:
    """Compute the DTW alignment index paths between consecutive rows of the crochet graph.

    Unlike ``get_crochet_graph``, this returns raw index correspondences rather than 3D edge
    arrays — useful to generate instructions from the column edges.

    Args:
        mesh: the cut mesh returned by ``compute_row_column_order``
        row_order ((v,), float): geodesic-distance field (u-axis) on the cut mesh vertices
        column_order ((v,), float): tangent scalar field (v-axis) on the cut mesh vertices
        stitch_size (float): uniform sampling spacing in UV space; controls stitch density

    Returns:
        A list of length ``n_rows - 1``.  Each entry is an ``(m, 2)`` int array whose rows are
        ``(i, j)`` index pairs produced by FastDTW, meaning vertex ``i`` in row ``k`` is aligned
        to vertex ``j`` in row ``k+1``.
    """
    row_separated_v = calculate_crochet_graph_vertices(mesh, row_order, column_order, stitch_size)

    connectivity = [np.array(fastdtw(r1, r2, dist=euclidean)[1]) for r1, r2 in
                    zip(row_separated_v[:-1], row_separated_v[1:])]

    return connectivity


def get_crochet_graph(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                      stitch_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the full crochet graph: stitch vertices, row edges, and column edges.

    Args:
        mesh: the cut mesh returned by ``compute_row_column_order``
        row_order ((v,), float): geodesic-distance field (u-axis) on the cut mesh vertices
        column_order ((v,), float): tangent scalar field (v-axis) on the cut mesh vertices
        stitch_size (float): uniform sampling spacing in UV space; controls stitch density

    Returns:
        A 3-tuple ``(vertices, row_edges, column_edges)``:
            - vertices ((n, 3), float): all sampled stitch locations in 3D, concatenated across rows
            - row_edges ((m, 2, 3), float): pairs of adjacent stitch positions within each row;
              the last pair in each row wraps around to close the loop
            - column_edges ((p, 2, 3), float): DTW-aligned stitch pairs between consecutive rows,
              encoding increases and decreases where row lengths differ
    """
    row_separated_vertices = calculate_crochet_graph_vertices(mesh, row_order, column_order, stitch_size)

    vertices = np.concatenate(row_separated_vertices)

    row_edges = np.concatenate([np.stack([row, np.roll(row, -1, axis=0)], axis=1) for row in row_separated_vertices],
                               axis=0)

    column_edges = np.concatenate([get_column_edges(r1, r2) for r1, r2 in
                                   zip(row_separated_vertices[:-1], row_separated_vertices[1:])], axis=0)

    return vertices, row_edges, column_edges


def get_column_edges(row1: np.ndarray, row2: np.ndarray) -> np.ndarray:
    """Compute column edges between two adjacent rows using FastDTW alignment.

    DTW handles rows of unequal length, so one stitch in ``row1`` can match multiple nodes in
    ``row2`` (increase) or vice-versa (decrease).

    Args:
        row1 ((n, 3), float): 3D stitch positions in the earlier row
        row2 ((m, 3), float): 3D stitch positions in the later row

    Returns:
        ((p, 2, 3), float): edge array where each entry ``[i]`` is a pair
        ``[row1[a], row2[b]]`` for the DTW-matched indices ``(a, b)``
    """
    _, path = fastdtw(row1, row2, dist=euclidean)
    path = np.array(path)
    return np.stack([row1[path[:, 0]], row2[path[:, 1]]], axis=1)


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
