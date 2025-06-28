import numpy as np

from src.utils import least_squares_with_equality


def test_least_squares_with_equality_basic_case():
    """ test least_squares_with_equality for a basic case """
    A = np.array([[1, 0], [0, 1]])
    c = np.array([3, 4])
    B = np.array([[1, -1]])  # x1 - x2 = 0 --> x1 = x2

    x = least_squares_with_equality(A, c, B)
    assert np.allclose(x[0], x[1]), "Constraint x1 = x2 not satisfied"
    expected = np.array([(3 + 4) / 2, (3 + 4) / 2])
    assert np.allclose(x, expected, atol=1e-6)


def test_least_squares_with_equality_no_constraint():
    """ test least_squares_with_equality for a no constraint """
    A = np.array([[2, 0], [0, 1]])
    c = np.array([2, 1])
    B = np.zeros((0, 2))  # No constraint

    x = least_squares_with_equality(A, c, B)
    expected = np.linalg.lstsq(A, c, rcond=None)[0]
    assert np.allclose(x, expected, atol=1e-6)


def test_least_squares_with_equality_multiple_constraints():
    """ test least_squares_with_equality for multiple constraints """
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c = np.array([1.0, 2.0, 3.0])
    B = np.array([
        [1, 0, -1],  # x1 = x3
        [0, 1, -1]  # x2 = x3
    ])

    x = least_squares_with_equality(A, c, B)
    # All values should be the average of c
    avg = np.mean(c)
    assert np.allclose(x, [avg, avg, avg], atol=1e-6)

