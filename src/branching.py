import numpy as np
import trimesh as tm


def find_saddle_points(mesh: tm.Trimesh, scalar_field: np.ndarray) -> np.ndarray:
    """ Find the saddle points of a scalar field defined on the vertices of a mesh.

    Following the discrete Morse theory classification, a vertex is classified by the number of times the field
    alternates between higher and lower than the vertex itself while walking around its link:
        - 0 changes: a local minimum or a local maximum
        - 2 changes: a regular point
        - more than 2 changes: a saddle point (a monkey saddle has 6, and so on)

    Saddle points are where the isolines of the field split or merge, so they mark the places the crochet graph has to
    branch.

    Args:
        mesh: a Trimesh object
        scalar_field ((v,), float): a scalar value per vertex of the mesh

    Returns:
        ((v,), bool) a mask that is True for every vertex that is a saddle point of the field
    """
    if scalar_field.shape != (len(mesh.vertices),):
        raise ValueError(f"Expected one scalar value per vertex - an array of shape ({len(mesh.vertices)},), "
                         f"received an array of shape {scalar_field.shape}")

    sign_change_count = get_vertex_sign_changes(mesh, scalar_field)

    return sign_change_count > 2


def get_vertex_sign_changes(mesh: tm.Trimesh, scalar_field: np.ndarray) -> np.ndarray:
    """ Count, for every vertex, how many times the scalar field changes sign around its link.

    The link of a vertex is the closed loop of edges opposite to it in its incident faces. Walking that loop, the count
    is the number of times the field switches from being above the vertex value to below it (or the other way around).
    Instead of ordering the link explicitly, every face is visited once from each of its three corners: for the corner
    `vertex` of a face, the opposite edge (`edge_start`, `edge_end`) is one segment of that link, and it contributes a
    change when its two endpoints lie on opposite sides of `vertex`.

    The comparison is done on the rank of the field rather than on its raw values, so that vertices with an equal field
    value are still strictly ordered and never produce a zero sign (see `_get_vertex_rank`).

    Expects a closed mesh, so that every link is a full loop and every count comes out even.

    Args:
        mesh: a Trimesh object
        scalar_field ((v,), float): a scalar value per vertex of the mesh

    Returns:
        ((v,), int) the number of sign changes around the link of each vertex; always an even number
    """
    num_vertices = len(mesh.vertices)

    rank = _get_vertex_rank(scalar_field)

    sign_change_count = np.zeros(num_vertices, dtype=np.int32)
    for i in range(3):
        vertex, edge_start, edge_end = mesh.faces[:, i], mesh.faces[:, (i + 1) % 3], mesh.faces[:, (i + 2) % 3]
        change = np.sign(rank[edge_start] - rank[vertex]) != np.sign(rank[edge_end] - rank[vertex])
        sign_change_count += np.bincount(vertex, weights=change, minlength=num_vertices).astype(np.int32)

    return sign_change_count


def _get_vertex_rank(scalar_field: np.ndarray) -> np.ndarray:
    """ Replace the values of a scalar field by their rank, breaking ties by the vertex index.

    This turns the field into a strict total order on the vertices: no two vertices share a value, so comparing two
    neighbors can never give 0. Plateaus of the original field are resolved consistently by index, which keeps the
    sign-change count of `get_vertex_sign_changes` well-defined.

    The order imposed on a plateau is arbitrary rather than geometric, so a field that is flat over a whole region will
    report saddles inside it that carry no meaning.

    Args:
        scalar_field ((v,), float): a scalar value per vertex of the mesh

    Returns:
        ((v,), int) the rank of each vertex, a permutation of 0..v-1 sorted by the field value
    """
    num_vertices = len(scalar_field)
    order = np.lexsort((np.arange(num_vertices), scalar_field))
    rank = np.empty(num_vertices, dtype=np.int64)
    rank[order] = np.arange(num_vertices)

    return rank
