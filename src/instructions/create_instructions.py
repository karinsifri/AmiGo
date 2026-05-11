from itertools import groupby

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
    if not isinstance(dtw_path, np.ndarray) or dtw_path.ndim != 2 or dtw_path.shape[1] != 2:
        raise ValueError("dtw_path must be a 2D array with shape (n, 2)")

    step_diffs = np.diff(dtw_path, axis=0)

    if set(step_diffs.ravel().tolist()) - {0, 1}:  # each component must be 0 or 1
        raise ValueError("Invalid Stitch")

    # encode each diff as a digit: [0,1]=1 (inc), [1,0]=2 (dec), [1,1]=3 (stitch start)
    encoded_steps = "".join((step_diffs[:, 0] * 2 + step_diffs[:, 1]).astype(str))

    # each stitch starts with a [1,1] step (encoded as "3"); split on it to get each stitch's payload
    stitch_payloads = encoded_steps.split("3")

    stitches = []
    for payload in stitch_payloads:
        if set(payload) - {'1', '2'} or ('1' in payload and '2' in payload):  # mixed inc+dec is invalid
            raise ValueError("Invalid Stitch")
        elif not payload:
            stitches.append("sc")
        elif "1" in payload:
            extra_inc_count = payload.count('1')
            stitches.append("inc" if extra_inc_count == 1 else f"inc{extra_inc_count}")
        elif "2" in payload:
            extra_dec_count = payload.count('2')
            stitches.append("dec" if extra_dec_count == 1 else f"dec{extra_dec_count}")

    return stitches


def create_final_instructions(rows_instructions: list[str]) -> str:
    """Format a per-row instruction list into a human-readable string, collapsing consecutive
    identical rows into a range.

    Args:
        rows_instructions: one instruction string per row, in order

    Returns:
        A formatted string where runs of identical instructions are written as
        ``"rows <start>-<end>: <instruction>"`` and singletons as ``"row <i>: <instruction>"``.
    """
    lines = []
    start_row = 0
    for instruction, group in groupby(rows_instructions):
        run_length = sum(1 for _ in group)
        end_row = start_row + run_length - 1
        if run_length == 1:
            lines.append(f"row {start_row}: {instruction}")
        else:
            lines.append(f"rows {start_row}-{end_row}: {instruction}")
        start_row += run_length
    return "\n".join(lines)
