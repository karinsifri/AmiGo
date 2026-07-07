import numpy as np
import pytest

from instructions.create_instructions import dtw_to_stitches, create_final_instructions


@pytest.mark.parametrize('dtw, stitches', [
    (np.array([[0, 0], [1, 1], [2, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
               [10, 12], [11, 13], [12, 14], [13, 15], [14, 16], [15, 17], [16, 17], [17, 17], [18, 17], [19, 18],
               [20, 19], [21, 20], [22, 21], [23, 22], [24, 23], [24, 24], [25, 25], [26, 26]]),
     ['sc', 'sc', 'inc', 'sc', 'sc', 'sc', 'inc', 'sc', 'sc', 'sc', 'sc', 'sc', 'sc', 'sc', 'sc', 'dec3', 'sc',
      'sc', 'sc', 'sc', 'sc', 'inc', 'sc', 'sc']
     ),
    (np.array([[0, 0], [1, 1]]), ['sc', 'sc'])
])
def test_dtw_to_stitches(dtw: np.ndarray, stitches: list[str]) -> None:
    """Tests the dtw_to_stitches function for a valid DTW path.

    Verifies that a realistic DTW alignment path between two crochet graph rows is correctly
    translated into a sequence of stitch instructions, including 'sc', 'inc', and 'dec' stitches.
    """
    assert dtw_to_stitches(dtw) == stitches


@pytest.mark.parametrize('dtw, creases, stitches', [
    (np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9], [11, 10],
               [12, 11], [13, 12], [14, 13], [15, 14], [16, 15], [17, 16], [18, 17], [19, 18], [20, 19], [21, 20],
               [22, 21], [23, 22], [24, 23], [25, 24], [26, 25], [27, 26], [28, 27], [29, 28], [30, 29], [31, 30],
               [32, 31], [33, 32], [34, 33], [35, 34], [36, 35], [37, 36], [38, 37]]),
     np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 0]),
     ['BLO sc', 'BLO sc', 'BLO sc', 'BLO dec', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc',
      'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'sc',
      'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc',
      'BLO sc', 'BLO sc', 'BLO sc', 'BLO sc', 'sc']),
    (np.array([[0, 0], [1, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 8], [7, 9], [7, 10], [8, 11],
               [8, 12]]),
     np.array([0, 0, 0, 0, 0, 0, -1, -1, 0]),
     ['sc', 'inc', 'sc', 'sc', 'sc', 'inc', 'FLO sc', 'FLO inc', 'inc']),
    # BLO on the first stitch — crease of node 0 goes into the preamble before the first "3"
    (np.array([[0, 0], [1, 1], [2, 2]]),
     np.array([1, 0, 0]),
     ['BLO sc', 'sc', 'sc']),
    # BLO on the last stitch
    (np.array([[0, 0], [1, 1], [2, 2]]),
     np.array([0, 0, 1]),
     ['sc', 'sc', 'BLO sc']),
    # all zeros — identical result to passing no creases
    (np.array([[0, 0], [1, 1], [2, 2]]),
     np.array([0, 0, 0]),
     ['sc', 'sc', 'sc']),
    # BLO with inc — inc step keeps the earlier-row vertex fixed, so the BLO label repeats in the payload
    (np.array([[0, 0], [1, 1], [1, 2], [2, 3]]),
     np.array([0, 1, 0]),
     ['sc', 'BLO inc', 'sc']),
    # FLO with dec — dec steps advance the earlier row; both dec vertices are FLO
    (np.array([[0, 0], [1, 1], [2, 1], [3, 2]]),
     np.array([0, -1, -1, 0]),
     ['sc', 'FLO dec', 'sc']),
    # all FLO row
    (np.array([[0, 0], [1, 1], [2, 2]]),
     np.array([-1, -1, -1]),
     ['FLO sc', 'FLO sc', 'FLO sc']),
])
def test_dtw_to_stitches_with_creases(dtw, creases, stitches):
    """Tests that BLO/FLO crease labels are correctly applied to each stitch type.

    Covers: BLO/FLO on the first and last stitch, all-zeros creases (no prefix),
    BLO with inc, FLO with dec, and an all-FLO row.
    """
    assert dtw_to_stitches(dtw, creases) == stitches


@pytest.mark.parametrize('dtw', [
    np.array([[0, 0], [1, 0], [1, 1]]),  # mixed stitch
    np.array([[0, 0], [2, 0]]),  # invalid path
    np.array([[0, 0], [0, 2]]),  # invalid path
    np.array([[0, 0], [-1, 0]]),
    np.array([[0, 0], [0, 0], [1, 1]])  # stationary step
])
def test_dtw_to_stitches_case_failed(dtw: np.ndarray) -> None:
    """Tests that dtw_to_stitches raises ValueError for invalid DTW paths.

    Ensures that paths containing steps larger than 1, negative steps, stationary steps, or
    mixed inc+dec within a single stitch segment all raise a ValueError with the message
    'Invalid Stitch'.
    """
    with pytest.raises(ValueError) as error_info:
        dtw_to_stitches(dtw)
    assert str(error_info.value) == 'Invalid Stitch'


@pytest.mark.parametrize('dtw', [
    np.array([0, 1, 2]),  # 1D array
    np.array([[0, 0, 0], [1, 1, 1]]),  # 3 columns
    np.zeros((3, 1)),  # 1 column
    [[0, 0], [1, 1]],  # list, not ndarray
])
def test_dtw_to_stitches_invalid_shape(dtw):
    """Tests that dtw_to_stitches raises ValueError for inputs with the wrong shape or type."""
    with pytest.raises(ValueError, match="dtw_path must be a 2D array with shape"):
        dtw_to_stitches(dtw)


@pytest.mark.parametrize('creases', [
    [0, 1, -1],                     # list, not ndarray
    np.array([[0, 1], [-1, 0]]),     # 2D array
    np.array([2, 0, 1]),             # value outside {-1, 0, 1}
    np.array([-2, 0, 1]),            # value outside {-1, 0, 1}
])
def test_dtw_to_stitches_invalid_creases(creases):
    """Tests that dtw_to_stitches raises ValueError when creases has wrong type, shape, or values."""
    dtw = np.array([[0, 0], [1, 1], [2, 2]])
    with pytest.raises(ValueError, match="creases must be a vector"):
        dtw_to_stitches(dtw, creases)


@pytest.mark.parametrize('rows, expected_res', [
    (['a', 'a', 'a', 'b', 'a'], 'rows 0-2: a\nrow 3: b\nrow 4: a'),
    (['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
     'row 0: a\nrow 1: b\nrow 2: c\nrow 3: a\nrow 4: b\nrow 5: c\nrow 6: a\nrow 7: b\nrow 8: c'),
    (['a', 'a', 'b', 'b', 'c', 'c', 'c', 'a', 'b', 'c'],
     'rows 0-1: a\nrows 2-3: b\nrows 4-6: c\nrow 7: a\nrow 8: b\nrow 9: c'),
])
def test_create_final_instructions(rows: list[str], expected_res: str) -> None:
    """Tests the create_final_instructions function for various row instruction sequences.

    Verifies that consecutive identical instructions are collapsed into a range format
    ('rows X-Y: ...'), while non-repeated instructions are each printed individually
    ('row X: ...').
    """
    assert create_final_instructions(rows) == expected_res
