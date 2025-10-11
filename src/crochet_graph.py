import numpy as np
import trimesh as tm
import gpytoolbox as gpy
from potpourri3d import MeshHeatMethodDistanceSolver, EdgeFlipGeodesicSolver

from src.consts import HEAT_COEFFICIENT
from src.utils import least_squares_with_equality


def create_crochet_graph(mesh: tm.Trimesh, origin: int):
    v = mesh.vertices
    f = mesh.faces
    distance_solver = MeshHeatMethodDistanceSolver(v, f, t_coef=HEAT_COEFFICIENT)
    distance_field = distance_solver.compute_distance(origin)
    path_solver = EdgeFlipGeodesicSolver(v, f)
    geodesic_path = path_solver.find_geodesic_path(origin, np.argmax(distance_field))
    cut_path = get_path_cut(mesh, distance_field, geodesic_path)
    f_new, v_ind_new = gpy.cut_edges(f, cut_path)
    v_new = v[v_ind_new]
    row_order = distance_field[v_ind_new]
    cut_mesh = tm.Trimesh(vertices=v_new, faces=f_new, process=False)


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
    # initialize the returned array
    found_edges = np.full((len(points), 2), -1, dtype=np.int16)

    # get the face that the point lies on (or the closest face)
    _, _, relevant_face_idx = tm.proximity.closest_point(mesh, points)
    relevant_faces = mesh.faces[relevant_face_idx]

    # check if the point is one of the vertices of the face
    close_to_vertex = np.all(np.isclose(mesh.vertices[relevant_faces], points[:, None]), axis=-1)
    # returns two lists of indices, the first describes the points in the given list that are close to points,
    # the second describes which of the face vertices is the point close to
    point_is_vertex = np.where(close_to_vertex)
    # fill the returned array with the vertices indices found to be close to the points
    found_edges[point_is_vertex[0], :] = np.tile(relevant_faces[point_is_vertex[0], point_is_vertex[1]], (2, 1)).T

    # check if the point is on one of the edges in the faces
    edge_vectors = mesh.vertices[np.roll(relevant_faces, -1, axis=1)] - mesh.vertices[relevant_faces]
    vertex_to_point_vectors = points[:, None] - mesh.vertices[relevant_faces]
    # a point x is on a line between a and b if the vector ab is linearly dependent on the vector ax
    linearly_dependant = np.linalg.matrix_rank(np.stack([edge_vectors, vertex_to_point_vectors], axis=2), tol=1e-10) == 1
    point_on_edge = np.logical_and(linearly_dependant, ~np.any(close_to_vertex, axis=-1)[:, None])
    # fill the returned array with the edge indices found
    found_edges[point_on_edge.any(axis=1)] = np.vstack([relevant_faces[point_on_edge],
                                                  np.roll(relevant_faces, -1, axis=1)[point_on_edge]]).T

    return found_edges


def get_path_cut(mesh: tm.Trimesh, distance_field: np.ndarray, path: np.ndarray) -> np.ndarray:
    """ Given a mesh, distance-field and a geodesic path, find the vertices of a path to cut the mesh by.
    TODO: write documentation

    Args:
        mesh: a Trimesh object
        distance_field ((v, ), float): a distance field from the seed point
        path ((n, 3), float): the geodesic path from the seed point to the distance-field maximum where the points lay
            on the mesh edges

    Returns:
        ((n, 2), int) a list of edges of the cut path
    """
    path_edges = find_edges_from_points(mesh, path)
    _, _, face_idx = tm.proximity.closest_point(mesh, path)
    edge_vectors = mesh.vertices[path_edges[:, 0]] - mesh.vertices[path_edges[:, 1]]
    gradients = np.reshape(gpy.grad(mesh.vertices, mesh.faces) @ distance_field, (-1, 3), order='F')
    rotated = np.cross(mesh.face_normals, gradients, axis=1)
    dot_products = np.sum(edge_vectors * rotated[face_idx], axis=1)
    # select the "right" side of the edge
    # where the angle between the rotated gradient and the edge is in the range [-90, 90]
    cut_vertices = path_edges[np.arange(path_edges.shape[0]), (dot_products >= 0).astype(np.uint8)]
    # convert the vertex trail to a connected path of edges
    cut_edges = np.vstack([cut_vertices, np.roll(cut_vertices, -1)]).T
    cut_edges = cut_edges[cut_edges[:, 0] != cut_edges[:, 1]]  # remove degenerate edges
    return cut_edges


def get_column_order(mesh: tm.Trimesh, distance_field: np.array, geodesic_path: np.array) -> np.ndarray:
    # TODO: refactor, add documentation
    grad = gpy.grad(mesh.vertices, mesh.faces)
    grad_operator = grad.toarray().reshape((len(mesh.faces), 3, len(mesh.faces)), order='F')
    distance_gradient = (grad @ distance_field).reshape((-1, 3), order='F')
    rotated_gradient = np.cross(mesh.face_normals, distance_gradient, axis=1)
    A = np.sum(rotated_distance[:, :, None] * grad_operator, axis=1)
    condition_edges = find_edges_from_points(mesh, geodesic_path)
    B = np.zeros((len(condition_edges), len(new_v)))
    B[np.arange(len(condition_edges)), condition_edges[:, 0]] = np.linalg.norm(
        new_v[condition_edges[:, 1]] - geodesic_path, axis=-1)
    B[np.arange(len(condition_edges)), condition_edges[:, 1]] = np.linalg.norm(
        new_v[condition_edges[:, 0]] - geodesic_path, axis=-1)
    zero_vert = np.argwhere(condition_edges[:, 0] == condition_edges[:, 1])
    B[zero_vert, condition_edges[zero_vert, 0]] = 1
    return least_squares_with_equality(A, np.ones(len(mesh.faces,)), B)
