from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class Program:
    """Represents a crochet program: a sequence of instructions (strings) or loops.

    This class provides methods to fold raw instructions into a compact, human-readable form using loop folding, as
    described in Section 5.2 of the AmiGo paper (Program to Human-readable Pattern).

    Attributes:
        content (list[Loop | str]): The instructions in this program. Each element is either a raw instruction string
            or a Loop object representing repetitions.
    """

    content: list[Loop | str]

    def __init__(self, content: list[Loop | str]) -> None:
        """Initializes a Program from a sequence of instructions or loops.

        Args:
            content (list[Loop | str]): A list of instructions or Loop objects.
        """
        self.content = content

    @classmethod
    def fold_instructions(cls, instructions: list[str]) -> Program:
        """Folds a flat sequence of crochet instructions into a compact Program.

        The folding process detects maximal repeated subsequences of instructions and groups them into Loop objects.
        Folding proceeds recursively to ensure nested repetitions are captured.

        Args:
            instructions (list[str]): A flat list of crochet instructions.

        Returns:
            Program: A program containing folded loops and/or raw instructions.
        """
        n = len(instructions)

        for pattern_size in range(n // 2, 0, -1):
            pattern_start = 0

            while pattern_start <= n - 2 * pattern_size:
                pattern = instructions[pattern_start:pattern_start + pattern_size]
                tail = instructions[pattern_start + pattern_size:]
                loop, rest = find_loop(pattern, tail)

                if loop is not None:
                    head = instructions[:pattern_start]
                    return Program.fold_instructions(head).extend(
                        loop.extend(Program.fold_instructions(rest))
                    )

                pattern_start += 1

        return Program(instructions)

    def __len__(self) -> int:
        """ Returns the number of top-level elements in this program. """
        return len(self.content)

    def __str__(self) -> str:
        """ Returns a human-readable string representation of the program. """
        return ",".join(map(str, self.content))

    def __eq__(self, other: object) -> bool:
        """ Checks equality by comparing string representations. """
        return str(self) == str(other)

    def __getitem__(self, index: int) -> Loop | str:
        """ Retrieves the element at the given index. """
        return self.content[index]

    def extend(self, other: Program) -> Program:
        """ Concatenates this program with another. """
        return Program(self.content + other.content)


@dataclass
class Loop:
    """Represents a repeated sequence of crochet instructions.

    Attributes:
        num_repetitions (int): The number of times the loop content repeats.
        content (Program): The program content that is repeated.
    """

    num_repetitions: int
    content: Program

    def __str__(self) -> str:
        """Returns a human-readable string representation of the loop.

        - If the loop contains a single element, returns in crochet notation, e.g., "3sc".
        - Otherwise, returns a bracketed form, e.g., "3*[sc,inc]".

        Returns:
            str: The string representation of the loop.
        """
        if len(self.content) == 1:
            return f"{self.num_repetitions}{self.content}"
        return f"{self.num_repetitions}*[{self.content}]"

    def extend(self, program: Program) -> Program:
        """Merges this loop with the beginning of another program if possible.

        This supports folding adjacent repetitions into a larger loop.

        Args:
            program (Program): The program to merge with.

        Returns:
            Program: A new Program that combines this loop with the given program.
        """
        tail_content = program.content
        if len(program) > 0 and isinstance(program[0], Loop) and program[0].content == self.content:
            self.num_repetitions += program[0].num_repetitions
            tail_content = program[1:]
        elif self.content == Program(program[:len(self.content)]):
            self.num_repetitions += 1
            tail_content = program[len(self.content):]
        return Program([self] + tail_content)

    @classmethod
    def fold(cls, num_repetitions: int, instructions: list[str]) -> Loop:
        """Folds repeated instructions into a Loop object.

        This method recursively folds the loop body before creating the Loop,
        ensuring nested repetitions are captured.

        Args:
            num_repetitions (int): Number of times the pattern repeats.
            instructions (list[str]): The repeated pattern.

        Returns:
            Loop: A Loop object representing the folded pattern.
        """
        content = Program.fold_instructions(instructions)
        if len(content) == 1 and isinstance(content[0], Loop):
            return Loop(num_repetitions * content[0].num_repetitions, content[0].content)
        return Loop(num_repetitions, content)


def find_loop(pattern: list[str], rest: list[str]) -> tuple[Optional[Loop], list[str]]:
    """Detects and folds repetitions of a given pattern at the start of `rest`.

    Args:
        pattern (list[str]): Candidate repeated subsequence.
        rest (list[str]): The tail sequence following the candidate.

    Returns:
        tuple[Optional[Loop], list[str]]:
            - If repetitions are found, returns (Loop, remaining_rest).
            - Otherwise, returns (None, pattern + rest).
    """
    num_repetitions = 1
    pattern_size = len(pattern)
    while len(rest) >= pattern_size:
        if not rest[:pattern_size] == pattern:
            break
        num_repetitions += 1
        rest = rest[pattern_size:]
    if num_repetitions == 1:
        return None, pattern + rest
    return Loop.fold(num_repetitions, pattern), rest
