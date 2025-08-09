from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Instructions:
    content: list[Loop | str]

    def __len__(self) -> int:
        return len(self.content)

    def __str__(self) -> str:
        return ",".join(map(str, self.content))


@dataclass
class Loop:
    num_repetitions: int
    content: Instructions

    def __str__(self) -> str:
        if len(self.content) == 1:
            return f"{self.num_repetitions}{self.content}"
        return f"{self.num_repetitions}*[{self.content}]"


def fold_loops():
    pass
