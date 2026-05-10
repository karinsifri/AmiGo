import numpy as np
import pytest

from instructions.create_instructions import dtw_to_stitches


@pytest.mark.parametrize('dtw, stitches', [
        (np.array([[0, 0], [1, 1], [2, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                  [10, 12], [11, 13], [12, 14], [13, 15], [14, 16], [15, 17], [16, 17], [17, 17], [18, 17], [19, 18],
                  [20, 19], [21, 20], [22, 21], [23, 22], [24, 23], [24, 24], [25, 25], [26, 26]]),
         ['sc', 'sc', 'inc', 'sc', 'sc', 'sc', 'inc', 'sc', 'sc', 'sc', 'sc', 'sc', 'sc', 'sc', 'sc', 'dec3', 'sc',
          'sc', 'sc', 'sc', 'sc', 'inc', 'sc', 'sc']
         )
])
def test_dtw_to_stitches(dtw: np.ndarray, stitches: list[str]) -> None:
    assert dtw_to_stitches(dtw) == stitches
