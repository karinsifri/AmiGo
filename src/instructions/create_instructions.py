import numpy as np


def dtw_to_stitches(arr: np.ndarray) -> list[str]:
    diff = np.diff(arr, axis=0)
    con_type_str = "".join((diff[:, 0] * 2 + diff[:, 1]).astype(str))
    raw_stitches = con_type_str.split("3")
    stitches = []
    for s in raw_stitches:
        if not set(s) <= {'1', '2'}:
            raise ValueError("Invalid Stitch")
        elif not s:
            stitches.append("sc")
        elif "1" in s:
            stitches.append(f"inc{s.count('1')}".replace("1", ""))
        elif "2" in s:
            stitches.append(f"dec{s.count('2')}".replace("1", ""))
    return stitches
