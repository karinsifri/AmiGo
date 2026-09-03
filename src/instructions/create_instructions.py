from itertools import groupby
from typing import Optional

import numpy as np

from src.instructions.loop_folding import Program


def get_row_stitch_sequences(connectivity: list[np.ndarray], creases: list[np.ndarray]) -> list[list[str]]:
    """Convert per-row DTW connectivity into per-row stitch instruction sequences.

    Args:
        connectivity: sequence of per-row-pair DTW paths
        creases: sequence of per-row crease labels

    Returns:
        list[list[str]]: one stitch-instruction list per row; every other row is reversed (back-and-forth crochet)
            to avoid shape deviation from the slant of the stitches.
    """
    return [dtw_to_stitches(dtw, bflo)[::(-1) ** i] for i, (dtw, bflo) in
            enumerate(zip(connectivity, creases, strict=True))]


def get_folded_rows(row_stitches: list[str], connectivity: list[np.ndarray]) -> list[str]:
    """Fold each row's stitch sequence into a compact loop notation with a trailing stitch count.

    Args:
        row_stitches: one stitch-instruction list per row
        connectivity: sequence of per-row-pair DTW paths

    Returns:
        list[str]: one folded instruction string per row, where the trailing number is the number of loops after that
            row.
    """
    return [f"{Program.fold(stitch_seq)} ({row_conn[-1][1] + 1})" for stitch_seq, row_conn in
            zip(row_stitches, connectivity, strict=True)]


def dtw_to_stitches(dtw_path: np.ndarray, creases: Optional[np.ndarray] = None) -> list[str]:
    """Convert a DTW alignment path between two crochet rows into a list of stitch instructions.

    Each step in the path is encoded as a digit (d_prev * 2 + d_next):
        - [0, 1] → 1: only the later row advances  — part of an 'inc' stitch
        - [1, 0] → 2: only the earlier row advances — part of a 'dec' stitch
        - [1, 1] → 3: both rows advance             — starts a new stitch

    Stitches are separated by '3' in the encoded string; the first stitch has no leading '3'.
    Each stitch's crease label ('b' for BLO, 'f' for FLO) is appended after the step that arrives at that node.
    The first node's crease has no incoming step, so it is prepended before the first '3'.

    Args:
        dtw_path ((n, 2) int): DTW alignment path between two consecutive crochet rows;
            column 0 indexes the earlier row, column 1 indexes the later row
        creases ((k,) int8, optional): crease label per vertex in the earlier row —
            1 for BLO, -1 for FLO, 0 for regular; when provided, stitches are prefixed
            with 'BLO' or 'FLO' accordingly

    Returns:
        list[str]: one stitch instruction per node in the earlier row, e.g. 'sc', 'inc',
            'dec2', 'BLO sc', 'FLO inc'
    """
    if not isinstance(dtw_path, np.ndarray) or dtw_path.ndim != 2 or dtw_path.shape[1] != 2:
        raise ValueError("dtw_path must be a 2D array with shape (n, 2)")

    if creases is not None and (not isinstance(creases, np.ndarray) or creases.ndim != 1 or set(creases) - {-1, 0, 1}):
        raise ValueError("creases must be a vector containing the values -1, 0 and 1")

    if creases is not None and len(creases) != dtw_path[:, 0].max() + 1:
        raise ValueError("creases array must include a crease type for each previous row's node")

    step_diffs = np.diff(dtw_path, axis=0)

    if set(step_diffs.ravel().tolist()) - {0, 1}:  # each component must be 0 or 1
        raise ValueError("Invalid Stitch")

    # one crease marker per DTW node, keyed by the earlier-row vertex index at that node
    edge_type_str = np.full(len(dtw_path), "", dtype=np.str_)
    if creases is not None:
        edge_types = creases[dtw_path[:, 0]]
        edge_type_str[edge_types == 1] = 'b'
        edge_type_str[edge_types == -1] = 'f'

    # build the encoded string: the crease of node i is placed after the step that arrives at i,
    # so after splitting on "3" it lands at the start of node i's stitch payload.
    # node 0 has no incoming step, so its crease is prepended before the first "3".
    encoded_steps = edge_type_str[0] + "".join((step_diffs[:, 0] * 2 + step_diffs[:, 1]).astype(str)
                                               + edge_type_str[1:])

    # "3" separates consecutive stitches; the first stitch has no leading "3",
    # so split("3") yields exactly one payload per stitch
    stitch_payloads = encoded_steps.split("3")

    stitches = []
    for payload in stitch_payloads:
        if set(payload) - {'1', '2', 'b', 'f'} or ('1' in payload and '2' in payload):  # mixed inc+dec is invalid
            raise ValueError("Invalid Stitch")
        if "1" in payload:
            extra_inc_count = payload.count('1')
            s = ("inc" if extra_inc_count == 1 else f"inc{extra_inc_count}")
        elif "2" in payload:
            extra_dec_count = payload.count('2')
            s = ("dec" if extra_dec_count == 1 else f"dec{extra_dec_count}")
        else:
            s = "sc"

        if 'b' in payload:
            s = "BLO " + s
        if 'f' in payload:
            s = "FLO " + s

        stitches.append(s)

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
