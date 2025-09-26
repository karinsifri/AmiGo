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


def fold_loops(instructions: list[str]) -> list[Loop | str]:
    n = len(instructions)

    if n == 1:
        return instructions

    for pattern_size in range(n // 2, 0, -1):
        pattern_start = 0

        while pattern_start <= n - 2 * pattern_size:
            pattern = instructions[pattern_start:pattern_start + pattern_size]

            if len(set(pattern)) == 1:
                pattern_start += 1
                continue

            count = 1
            while (pattern_start + count * pattern_size + pattern_size) <= n and \
                    instructions[
                    pattern_start + count * pattern_size: pattern_start + (count + 1) * pattern_size] == pattern:
                count += 1

            if count > 1:
                folded = []
                # Fold BEFORE
                folded.extend(fold_loops(instructions[:pattern_start]))

                # Fold INSIDE (in case the pattern itself can be folded further)
                inner_folded = fold_loops(pattern)

                # Fold the loop
                if len(inner_folded) == 1 and isinstance(inner_folded[0], tuple):
                    folded.append((count * inner_folded[0][0], inner_folded[0][1]))
                else:
                    folded.append((count, inner_folded))

                # Fold AFTER
                folded.extend(fold_loops(instructions[pattern_start + count * pattern_size:]))
                return folded

            pattern_start += 1
    return instructions
