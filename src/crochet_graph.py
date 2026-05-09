import numpy as np
import trimesh as tm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def get_row_connectivity(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                         stitch_size: float) -> list[np.ndarray]:
    row_separated_v = calculate_crochet_graph_vertices(mesh, row_order, column_order, stitch_size)

    connectivity = [np.array(fastdtw(row_separated_v[idx], row_separated_v[idx + 1], dist=euclidean)[1]) for idx in
                    range(len(row_separated_v) - 1)]

    return connectivity


def get_crochet_graph(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                      stitch_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    row_separated_vertices = calculate_crochet_graph_vertices(mesh, row_order, column_order, stitch_size)

    vertices = np.concatenate(row_separated_vertices)

    row_edges = np.concatenate([np.stack([row, np.roll(row, -1, axis=0)], axis=1) for row in row_separated_vertices],
                               axis=0)

    column_edges = np.concatenate([get_column_edges(r1, r2) for r1, r2 in
                                   zip(row_separated_vertices[:-1], row_separated_vertices[1:])], axis=0)

    return vertices, row_edges, column_edges


def get_column_edges(row1: np.ndarray, row2: np.ndarray) -> np.ndarray:
    _, path = fastdtw(row1, row2, dist=euclidean)
    path = np.array(path)
    return np.stack([row1[path[:, 0]], row2[path[:, 1]]], axis=1)


def calculate_crochet_graph_vertices(mesh: tm.Trimesh, row_order: np.ndarray, column_order: np.ndarray,
                                     stitch_size: float) -> list[np.ndarray]:
    samples_u, samples_v = np.meshgrid(np.arange(0, row_order.max(), stitch_size),
                                       np.arange(0, column_order.max(), stitch_size))
    i_idx, j_idx = np.meshgrid(np.arange(samples_u.shape[1]), np.arange(samples_u.shape[0]))

    flatten_mesh = tm.Trimesh(np.hstack([row_order, column_order, np.zeros_like(row_order)]), mesh.faces)

    sampled_points = np.stack((np.append(samples_u, row_order.max()), np.append(samples_v, 0),
                               np.zeros((samples_u.size + 1,))), axis=-1)

    _, point_dist, face_id = tm.proximity.closest_point(flatten_mesh, sampled_points)
    sampled_points = sampled_points[point_dist < 1e-10]
    face_id = face_id[point_dist < 1e-10]

    sampled_baricentric = tm.triangles.points_to_barycentric(flatten_mesh.vertices[mesh.faces[face_id]],
                                                             sampled_points)

    sampled_3d = tm.triangles.barycentric_to_points(mesh.vertices[mesh.faces[face_id]], sampled_baricentric)

    left_i, left_j = np.append(i_idx, i_idx.max() + 1).ravel()[point_dist < 1e-10], np.append(j_idx, 0).ravel()[
        point_dist < 1e-10]

    row_separated = [sampled_3d[left_i == i] for i in np.arange(left_i.max() + 1)]

    return row_separated
