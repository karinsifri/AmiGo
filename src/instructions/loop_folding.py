from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class Program:
    content: list[Loop | str]

    def __init__(self, content: list[Loop | str]) -> None:
        self.content = content

    @classmethod
    def fold_instructions(cls, instructions: list[str]) -> Program:
        n = len(instructions)

        if n == 0:
            return Program(instructions)

        for pattern_size in range(n // 2, 0, -1):
            pattern_start = 0

            while pattern_start <= n - 2 * pattern_size:
                pattern = instructions[pattern_start:pattern_start + pattern_size]
                tail = instructions[pattern_start + pattern_size:]
                loop, rest = find_loop(pattern, tail)

                if loop is not None:
                    head = instructions[:pattern_start]
                    return Program.fold_instructions(head).extend(loop.extend(Program.fold_instructions(rest)))

                pattern_start += 1

        return Program(instructions)

    def __len__(self) -> int:
        return len(self.content)

    def __str__(self) -> str:
        return ",".join(map(str, self.content))

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __getitem__(self, index: int) -> Loop | str:
        return self.content[index]


@dataclass
class Loop:
    num_repetitions: int
    content: Program

    def __str__(self) -> str:
        if len(self.content) == 1:
            return f"{self.num_repetitions}{self.content}"
        return f"{self.num_repetitions}*[{self.content}]"

    def extend(self, program: Program) -> Program:
        tail_content = program.content
        if isinstance(program[0], Loop) and program[0].content == self.content:
            self.num_repetitions += program[0].num_repetitions
            tail_content = program[1:]
        if self.content == Program(program[:len(self.content)]):
            self.num_repetitions += 1
            tail_content = program[len(self.content):]
        return Program([self] + tail_content)


def find_loop(pattern: list[str], rest: list[str]) -> tuple[Optional[Loop], list[str]]:
    num_repetitions = 1
    pattern_size = len(pattern)
    while len(rest) >= pattern_size:
        if not rest[:pattern_size] == pattern:
            break
        num_repetitions += 1
        rest = rest[pattern_size:]
    if num_repetitions == 1:
        return None, pattern + rest
    return Loop(num_repetitions, Program.fold_instructions(pattern)), rest
