import numpy as np
import trimesh as tm

from src.crochet_graph import find_edges_from_points


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
    np.testing.assert_allclose(np.sort(expected_result, axis=1), np.sort(result, axis=1))
