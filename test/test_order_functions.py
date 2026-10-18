import numpy as np
import pytest
import trimesh as tm

from src.order_functions import find_edges_from_cut_mesh_points, find_edges_from_points, get_path_condition


@pytest.fixture
def cut_strip():
    """ A flat grid of four quads in the z=0 plane, already cut along its middle row of vertices.

    The three middle vertices are duplicated: 3-5 carry the lower strip (y < 1) and 6-8 carry the upper strip
    (y > 1), both at the positions (0, 1, 0), (1, 1, 0) and (2, 1, 0). Every face is wound counter-clockwise, so
    all the face normals are +z, and the shortest edge in the mesh is 1 long.

         9 ----- 10 ----- 11      y = 2
         |    /   |    /   |
         6 -----  7 -----  8      y = 1   (upper copies)
        - - - - - cut - - - - -
         3 -----  4 -----  5      y = 1   (lower copies)
         |    /   |    /   |
         0 -----  1 -----  2      y = 0
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],                # 0-2   bottom row
                         [0, 1, 0], [1, 1, 0], [2, 1, 0],                # 3-5   middle row, lower copies
                         [0, 1, 0], [1, 1, 0], [2, 1, 0],                # 6-8   middle row, upper copies
                         [0, 2, 0], [1, 2, 0], [2, 2, 0]], dtype=float)  # 9-11  top row
    faces = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],        # lower strip
                      [6, 7, 10], [6, 10, 9], [7, 8, 11], [7, 11, 10]])  # upper strip

    # process=False keeps trimesh from merging the duplicated vertices back together
    return tm.Trimesh(vertices=vertices, faces=faces, process=False)


def isoline_direction_towards(cut_strip: tm.Trimesh, direction: np.ndarray) -> np.ndarray:
    """ A per-face isoline direction all pointing at the given direction, as the side to receive the lower values. """
    return np.tile(direction, (len(cut_strip.faces), 1))


def test_find_edges_from_points():
    """ Test the find_edges_from_points function for known points on a simple mesh """
    mesh = tm.primitives.Sphere().to_mesh()
    points = np.array([[0.131200378813013, 0.484441642060668, 0.864929335863275],
                       [0.223649830102071, 0.412505675867459, 0.880088018944412],
                       [0.261548659773797, 0.381456611623779, 0.883848341446897],
                       [0.31629688748929, 0.333487701105145, 0.884647433821021],
                       [0.385613341666478, 0.269091897624113, 0.880502144638511],
                       [0.404690797585192, 0.250112667841953, 0.877676841380714],
                       [1, 1, 1],
                       [-0.131200378813013, -0.484441642060668, -0.864929335863275]])
    expected_result = np.array([[402, 402],
                                [108, 405],
                                [108, 419],
                                [419, 403],
                                [403, 416],
                                [31, 416],
                                [-1, -1],
                                [480, 480]])
    result = find_edges_from_points(mesh, points)
    np.testing.assert_array_equal(np.sort(expected_result, axis=1), np.sort(result, axis=1))


def test_get_path_condition():
    """ Test the get_path_condition function for a simple case """
    vertices = np.array([[3, 14, 31],
                         [29, 0, 55],
                         [37, 4, 73],
                         [79, 57, 89],
                         [47, 25, 15]])
    condition_edges = np.array([[0, 0],
                                [2, 1],
                                [3, 2]])
    path = np.array([[3, 14, 31],           # point is on a vertex
                     [35, 3, 68.5],         # point is on an edge
                     [70.6, 46.4, 85.8]])   # point is on an edge
    expected_result = np.array([[1, 0, 0, 0, 0],
                                [0, 5.024937810560445, 15.074813431681335, 0, 0],
                                [0, 0, 13.898201322473357, 55.5928052898934, 0]])
    actual_result = get_path_condition(vertices, condition_edges, path)
    np.testing.assert_allclose(expected_result, actual_result)


@pytest.mark.parametrize("direction, expected_result", [
    ([0, -1, 0], [[4, 4], [5, 5]]),  # the lower strip carries the copies 4 and 5
    ([0, 1, 0], [[7, 7], [8, 8]]),  # the upper strip carries the copies 7 and 8
    (np.array([1, -1, 0]) / np.sqrt(2), [[4, 4], [5, 5]]),  # a diagonal push lands inside a face of the lower strip
    (np.array([1, 1, 0]) / np.sqrt(2), [[7, 7], [8, 8]]),  # and inside a face of the upper strip
])
def test_find_edges_from_cut_mesh_points_picks_the_copy_the_direction_points_at(cut_strip, direction, expected_result):
    """ Points on the duplicated vertices of the cut should resolve to the copies on the side the isoline direction
    points at, so flipping the direction selects the other copy of the same vertices. """
    points = np.array([[1, 1, 0],
                       [2, 1, 0]])
    result = find_edges_from_cut_mesh_points(cut_strip, points, isoline_direction_towards(cut_strip, direction))
    np.testing.assert_array_equal(expected_result, result)


def test_find_edges_from_cut_mesh_points_keeps_the_unambiguous_points_in_place(cut_strip):
    """ A point on an edge and a point on a vertex that the cut did not duplicate must be returned as they are, and
    the resolved duplicated vertices must stay aligned with their own points. """
    points = np.array([[0.5, 0, 0],     # point is on an edge
                       [1, 0, 0],       # point is on a vertex that was not duplicated
                       [1, 1, 0],       # point is on a duplicated vertex
                       [2, 1, 0]])      # point is on a duplicated vertex
    expected_result = np.array([[0, 1],
                                [1, 1],
                                [4, 4],
                                [5, 5]])
    result = find_edges_from_cut_mesh_points(cut_strip, points, isoline_direction_towards(cut_strip, [0, -1, 0]))
    np.testing.assert_array_equal(np.sort(expected_result, axis=1), np.sort(result, axis=1))


def test_find_edges_from_cut_mesh_points_without_ambiguous_points(cut_strip):
    """ Points that are all unambiguous should be resolved without attempting to disambiguate any of them. """
    points = np.array([[0.5, 0, 0],
                       [1, 0, 0]])
    expected_result = np.array([[0, 1],
                                [1, 1]])
    result = find_edges_from_cut_mesh_points(cut_strip, points, isoline_direction_towards(cut_strip, [0, -1, 0]))
    np.testing.assert_array_equal(np.sort(expected_result, axis=1), np.sort(result, axis=1))


def test_find_edges_from_cut_mesh_points_raises_for_a_point_off_the_edges(cut_strip):
    """ A point that lies on no mesh edge or vertex cannot be constrained, and should be reported. """
    points = np.array([[0.5, 0.25, 0]])
    with pytest.raises(ValueError, match="not on a mesh edge or vertex"):
        find_edges_from_cut_mesh_points(cut_strip, points, isoline_direction_towards(cut_strip, [0, -1, 0]))
