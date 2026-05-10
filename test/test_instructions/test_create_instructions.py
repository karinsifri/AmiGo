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
