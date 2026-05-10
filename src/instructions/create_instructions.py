import numpy as np


def dtw_to_stitches(dtw_path: np.ndarray) -> list[str]:
    """ This function converts dtw path to crochet stitches. The conversion is based on the advancement of the path
    along both rows of the crochet graph:
        - If the previous row does not advance to the next node (diff [0, 1], binary 1) this edge will be part of an
            'inc' stitch (one old node fans into multiple new nodes)
        - If the next row does not advance to the next node (diff [1, 0], binary 2) this edge will be part of a 'dec'
            stitch (multiple old nodes collapse into one new node)
        - All stitches start with an advancement of both rows to the next node (diff [1, 1], binary 3). If there are no
            'inc' or 'dec' stitches after - this edge will represent a 'sc' stitch

    We will represent those edges as a string (where each diff is encoded as a digit: d_prev × 2 + d_next)
    and split the resulting string to individual stitches.

    Args:
        dtw_path: a dtw path along two consecutive rows of the crochet graph, where column 0 indexes
            the earlier row and column 1 indexes the later row

    Returns:
        a list of the stitches matching to the edges between the given crochet graph rows
    """
    # calculate differences
    diff = np.diff(dtw_path, axis=0)

    if set(diff.ravel().tolist()) - {0, 1}:  # check for invalid path
        raise ValueError("Invalid Stitch")

    # encode each diff as a digit: [0,1]=1 (inc), [1,0]=2 (dec), [1,1]=3 (stitch start)
    con_type_str = "".join((diff[:, 0] * 2 + diff[:, 1]).astype(str))

    # the start of every stitch is always an advancement of both rows
    raw_stitches = con_type_str.split("3")

    stitches = []
    for s in raw_stitches:
        if set(s) - {'1', '2'} or ('1' in s and '2' in s):  # mixed inc+dec within one stitch is invalid
            raise ValueError("Invalid Stitch")
        elif not s:  # empty string
            stitches.append("sc")
        elif "1" in s:  # string composed by 1's
            count = s.count('1')
            stitches.append("inc" if count == 1 else f"inc{count}")
        elif "2" in s:  # string composed by 2's
            count = s.count('2')
            stitches.append("dec" if count == 1 else f"dec{count}")

    return stitches


def create_final_instructions(rows_instructions: list[str]) -> str:
    prog = ''
    i = 0
    n = len(rows_instructions)

    while i < n:
        # Only fold flat instruction strings, not loops
        if isinstance(rows_instructions[i], str):
            j = i + 1
            while j < n and rows_instructions[j] == rows_instructions[i]:
                j += 1
            repeat_count = j - i
            if repeat_count > 1:
                prog += f"rows {i}-{j-1}: {rows_instructions[i]}\n"
            else:
                prog += f"row {i}: {rows_instructions[i]}\n"
            i = j
        else:
            prog += f"row {i}: {rows_instructions[i]}\n"
            i += 1

    return prog
