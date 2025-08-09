from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Program:
    content: list[Loop | str]

    def __len__(self) -> int:
        return len(self.content)

    def __str__(self) -> str:
        return ",".join(map(str, self.content))


@dataclass
class Loop:
    num_repetitions: int
    content: Program

    def __str__(self) -> str:
        if len(self.content) == 1:
            return f"{self.num_repetitions}{self.content}"
        return f"{self.num_repetitions}*[{self.content}]"


def fold_loops(instructions: list[str]) -> list[Loop|str]:
    n = len(instructions)

    if n == 1:
        return instructions

    for window_size in range(n // 2, 0, -1):
        i = 0
        while i <= n - 2 * window_size:
            pattern = instructions[i:i + window_size]
            if len(set(pattern)) == 1:
                i += 1
                continue

            count = 1
            while (i + count * window_size + window_size) <= n and \
                    instructions[i + count * window_size: i + (count + 1) * window_size] == pattern:
                count += 1

            if count > 1:
                folded = []
                # Fold BEFORE
                folded.extend(fold_loops(instructions[:i]))

                # Fold INSIDE (in case the pattern itself can be folded further)
                inner_folded = fold_loops(pattern)

                # Fold the loop
                if len(inner_folded) == 1 and isinstance(inner_folded[0], tuple):
                    folded.append((count * inner_folded[0][0], inner_folded[0][1]))
                else:
                    folded.append((count, inner_folded))

                # Fold AFTER
                folded.extend(fold_loops(instructions[i + count * window_size:]))
                return folded

            i += 1
    return instructions