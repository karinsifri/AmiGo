import gpytoolbox as gpy
import numpy as np
import trimesh as tm
from potpourri3d import MeshHeatMethodDistanceSolver, EdgeFlipGeodesicSolver

from src.consts import HEAT_COEFFICIENT, EPSILON
from src.utils import least_squares_with_equality


def compute_row_column_order(mesh: tm.Trimesh, origin: int) -> tuple[tm.Trimesh, np.ndarray, np.ndarray]:
    # TODO: documentation & refactor
    distance_solver = MeshHeatMethodDistanceSolver(mesh.vertices, mesh.faces, t_coef=HEAT_COEFFICIENT)
    distance_field = distance_solver.compute_distance(origin)
    path_solver = EdgeFlipGeodesicSolver(mesh.vertices, mesh.faces)
    geodesic_path = path_solver.find_geodesic_path(origin, np.argmax(distance_field))
    cut_path = get_path_cut(mesh, distance_field, geodesic_path)
    f_new, v_ind_new = gpy.cut_edges(mesh.faces, cut_path)
    v_new = mesh.vertices[v_ind_new]
    row_order = distance_field[v_ind_new]
    cut_mesh = tm.Trimesh(vertices=v_new, faces=f_new, process=False)
    column_order = get_column_order(cut_mesh, row_order, geodesic_path)
    return cut_mesh, row_order, column_order


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
    # initialize the returned array with the "not_found" value
    found_edges = np.full((len(points), 2), -1, dtype=np.int16)

    # get the face that the point lies on (or the closest face)
    _, _, relevant_face_idx = tm.proximity.closest_point(mesh, points)
    relevant_faces = mesh.faces[relevant_face_idx]

    # check if the point is one of the vertices of the face
    close_to_vertex = np.all(np.isclose(mesh.vertices[relevant_faces], points[:, None]), axis=-1)

    # returns two lists of indices, the first describes the points in the given list that are close to a vertex,
    # the second describes which of the face vertices is the point close to
    point_is_vertex = np.where(close_to_vertex)

    # get the index of the matching vertx in the full mesh
    matching_vertices = relevant_faces[point_is_vertex[0], point_is_vertex[1]]

    # fill the returned array with the vertices indices found to be close to the points
    found_edges[point_is_vertex[0], :] = np.tile(matching_vertices, (2, 1)).T

    # handles case: point is on one of the edges in the faces
    edge_vectors = mesh.vertices[np.roll(relevant_faces, -1, axis=1)] - mesh.vertices[relevant_faces]
    vertex_to_point_vectors = points[:, None] - mesh.vertices[relevant_faces]

    # a point x is on a line between a and b if the vector ab is linearly dependent on the vector ax
    linearly_dependant = np.linalg.matrix_rank(np.stack([edge_vectors, vertex_to_point_vectors], axis=2),
                                               tol=1e-10) == 1
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
    cut_edges = np.vstack([cut_vertices, np.roll(cut_vertices, -1)]).T[:-1]
    cut_edges = cut_edges[cut_edges[:, 0] != cut_edges[:, 1]]  # remove degenerate edges
    return cut_edges


def get_column_order(mesh: tm.Trimesh, distance_field: np.array, path: np.array) -> np.ndarray:
    """ Compute the column order function on a mesh - a tangent field with constraint of value 0 on the given path
    todo: documentation & cleanup

    Args:
        mesh: the cut mesh object to compute the column order function on
        distance_field: the field the column order should be tangent to
        path: points on the mesh that should receive the value 0 in the computed function

    Returns:
        ((v,), float) an array of the value of the g function on the given mesh
    """
    grad = gpy.grad(mesh.vertices, mesh.faces)
    grad_operator = grad.toarray().reshape((len(mesh.faces), 3, len(mesh.vertices)), order='F')
    distance_gradient = grad_operator @ distance_field
    rotated_gradient = np.cross(mesh.face_normals, distance_gradient, axis=1)
    A = np.sum(rotated_gradient[:, :, None] * grad_operator, axis=1)
    _, _, face_idx = tm.proximity.closest_point(mesh, path)
    condition_edges = find_edges_from_points(mesh,
                                             path + rotated_gradient[face_idx] * EPSILON)  # "push" to the zero side
    B = get_path_condition(mesh, condition_edges, path)
    return least_squares_with_equality(A, np.ones(len(mesh.faces, )), B)


def get_path_condition(vertices: np.ndarray, condition_edges: np.ndarray, path: np.ndarray) -> np.ndarray:
    """ Create a condition that makes sure that g(path)=0. the condition is a matrix B that should hold Bg=0.

    if the path point (c_i) lies on an edge (a_i, b_i), we will demand that the linear interpolation of the function
    on the edge vertices in the path point will be equal to 0.
        | c_i - a_i | * g(b_i) +  | c_i - b_i | * g(a_i) = 0
    Therefore, the constraint matrix should contain
        B_ij:   | c_i - b_i | where the j-th vertex is a_i and
                | c_i - a_i | where the j-th vertex is b_i

    if the path point (c_i) lies on a vertex, we will demand that the value of the function in this point will be equal
    to 0. Therefore, the constraint matrix should contain
        B_ij:   1 when c_i is the j-th vertex in the mesh

    Args:
        vertices ((v, 3), float): the vertices of the mesh
        condition_edges ((n, 2), int): the edges of the path condition
        path ((n, 3), float): the points on the mesh that should receive the value 0 in the computed function

    Returns:
        ((n, v), float) a condition matrix that makes sure that g(path)=0.
    """
    # initialize the condition matrix, there are n conditions ahd they should hold for all the points on the mesh
    condition_matrix = np.zeros((len(condition_edges), len(vertices)))

    # set the constraint for points that are on an edge
    condition_matrix[np.arange(len(condition_edges)), condition_edges[:, 0]] = np.linalg.norm(
        vertices[condition_edges[:, 1]] - path, axis=-1)
    condition_matrix[np.arange(len(condition_edges)), condition_edges[:, 1]] = np.linalg.norm(
        vertices[condition_edges[:, 0]] - path, axis=-1)

    # set the constraint for points that are on a vertex
    on_vertex = np.argwhere(condition_edges[:, 0] == condition_edges[:, 1])
    condition_matrix[on_vertex, condition_edges[on_vertex, 0]] = 1

    return condition_matrix
