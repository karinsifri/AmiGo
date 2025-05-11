import numpy as np
import trimesh as tm
from potpourri3d import MeshHeatMethodDistanceSolver, EdgeFlipGeodesicSolver

from src.consts import HEAT_COEFFICIENT


def create_crochet_graph(mesh: tm.Trimesh, origin: int):
    v = mesh.vertices
    f = mesh.faces
    distance_solver = MeshHeatMethodDistanceSolver(v, f, t_coef=HEAT_COEFFICIENT)
    row_order = distance_solver.compute_distance(origin)
    path_solver = EdgeFlipGeodesicSolver(v, f)
    geodesic_path = path_solver.find_geodesic_path(origin, np.argmax(row_order))


def find_edges_from_points(mesh: tm.Trimesh, points: np.ndarray) -> np.ndarray:
    """Given a list of points that lay on a mesh edges, find the edges they lay on.
        - If a point lies on a vertex, the returned edge will be [v, v] where v is the index of the vertex.
        - If no matching edge was found, the returned edge will be [-1, -1]

    Args:
        mesh: a Trimesh object
        points ((n, 3), float): a list points on the mesh edges

    Returns:
        ((n, 2), int) a list of edges the point lay on
    """
    edges = - np.ones((len(points), 2), dtype=np.int16)
    _, _, relevant_face_idx = tm.proximity.closest_point(mesh, points)
    relevant_faces = mesh.faces[relevant_face_idx]
    close_to_vertex = np.all(np.isclose(mesh.vertices[relevant_faces], points[:, None]), axis=-1)
    point_is_vertex = np.where(close_to_vertex)
    edges[point_is_vertex[0], :] = np.tile(relevant_faces[point_is_vertex[0], point_is_vertex[1]], (2, 1)).T
    face_edges = mesh.vertices[np.roll(relevant_faces, -1, axis=1)] - mesh.vertices[relevant_faces]
    vertex_to_point = points[:, None] - mesh.vertices[relevant_faces]
    linearly_dependant = np.linalg.matrix_rank(np.stack([face_edges, vertex_to_point], axis=2), tol=1e-10) == 1
    point_on_edge = np.logical_and(linearly_dependant, ~np.any(close_to_vertex, axis=-1)[:, None])
    edges[point_on_edge.any(axis=1)] = np.vstack([relevant_faces[point_on_edge],
                                      np.roll(relevant_faces, -1, axis=1)[point_on_edge]]).T
    return edges
