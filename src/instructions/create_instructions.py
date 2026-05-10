import numpy as np


def dtw_to_stitches(dtw_path: np.ndarray) -> list[str]:
    """ This function converts dtw path to crochet stitches. The conversion is based on the advancement of the path
    along both rows of the crochet graph:
        - If the previous row does not advance to the next node (diff [0, 1], binary 1) this edge will be part of an
            'inc' stitch (one old stitch fans into multiple new stitches)
        - If the next row does not advance to the next node (diff [1, 0], binary 2) this edge will be part of a 'dec'
            stitch (multiple old stitches collapse into one new stitch)
        - All stitches end with an advancement of both rows to the next node (diff [1, 1], binary 3). If there are no
            'inc' or 'dec' stitches before - this edge will represent a 'sc' stitch

    We will represent those edges as a string (where each diff is converted to its respective binary representation)
    and split the resulting string to individual stitches.

    Args:
        dtw_path: a dtw path along two consecutive rows of the crochet graph

    Returns:
        a list of the stitches matching to the edges between the given crochet graph rows
    """
    # calculate differences
    diff = np.diff(dtw_path, axis=0)

    # convert from binary representation
    con_type_str = "".join((diff[:, 0] * 2 + diff[:, 1]).astype(str))

    # the end of every stitch is always an advancement of both rows
    raw_stitches = con_type_str.split("3")

    stitches = []
    for s in raw_stitches:
        if set(s) - {'1', '2'}:  # mixed inc+dec within one stitch is invalid
            raise ValueError("Invalid Stitch")
        elif not s:             # empty string
            stitches.append("sc")
        elif "1" in s:          # string composed by 1's
            count = s.count('1')
            stitches.append("inc" if count == 1 else f"inc{count}")
        elif "2" in s:          # string composed by 2's
            count = s.count('2')
            stitches.append("dec" if count == 1 else f"dec{count}")

    return stitches
