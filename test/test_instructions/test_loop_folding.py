import pytest

from src.instructions.loop_folding import Program, Loop, _fold_repetitions


@pytest.mark.parametrize("instructions, folded_instructions", [
    ('a', 'a'),
    ('a a', '2a'),
    ('a a b a a b', '2*[2a,b]'),
    ('a b a b a b a b', '4*[a,b]'),
    ('a a a a a a a', '7a'),
    ('a a a a b c a b c c c', '3a,2*[a,b,c],2c'),
    ('sc inc sc sc sc inc sc sc sc inc sc sc', '3*[sc,inc,2sc]')
])
def test_instructions_folding(instructions: str, folded_instructions: str):
    """Tests the Program.fold method on flat instruction sequences.

    Ensures that raw sequences of crochet instructions are folded into the expected compact form with repetitions
    represented as loops.
    """
    actual = Program.fold(instructions.split())
    assert str(actual) == folded_instructions


@pytest.mark.parametrize("pattern, tail, expected_loop, expected_tail", [
    ('a', 'b', 'None', 'a b'),
    ('a', 'a b', '2a', 'b'),
    ('a b c', 'a b c a b c a b a c', '3*[a,b,c]', 'a b a c'),
    ('a b', 'a b a b a b a b a b', '6*[a,b]', ''),
    ('a b', 'a b a b a c b a b a b', '3*[a,b]', 'a c b a b a b'),
    ('a b', 'c a b a b a b a b a b', 'None', 'a b c a b a b a b a b a b')
])
def test__fold_repetitions(pattern: str, tail: str, expected_loop: str, expected_tail: str):
    """Tests the _fold_repetitions helper function.

    Verifies that given a candidate pattern and a sequence tail, the function correctly detects repeated subsequences,
    folds them into a Loop if possible, and returns the remaining instructions.
    """
    returned_loop, returned_tail = _fold_repetitions(pattern.split(), tail.split())
    assert str(returned_loop) == expected_loop and " ".join(returned_tail) == expected_tail


@pytest.mark.parametrize("loop, rest, expected_result", [
    (Loop(3, Program(['a'])), Program([Loop(2, Program(['a'])), 'b', 'a']), '5a,b,a'),
    (Loop(2, Program(['a', 'b'])), Program(['a', 'b', 'c']), '3*[a,b],c'),
    (Loop(2, Program(['a', 'b'])), Program(['c', 'a', 'b']), '2*[a,b],c,a,b'),
    (Loop(5, Program(['a'])), Program([]), '5a')
])
def test_loop_merge_with(loop: Loop, rest: Program, expected_result: str):
    """Tests the Loop.merge_with method.

    Ensures that loops correctly merge with subsequent programs when adjacent repetitions or compatible structures are
    found, producing a compact Program.
    """
    assert str(loop.merge_with(rest)) == expected_result


@pytest.mark.parametrize("program_1, program_2, expected_result", [
    (Program(['a']), Program(['b', 'a']), 'a,b,a'),
    (Program(['a', 'b', Loop(4, Program(['c', 'd', 'e'])), 'f', 'g', 'h']), Program(['a', Loop(3, Program(['b']))]),
     'a,b,4*[c,d,e],f,g,h,a,3b'),
    (Program(['h', 'e', 'l', 'l', 'o']), Program([]), 'h,e,l,l,o')
])
def test_program_concat(program_1: Program, program_2: Program, expected_result: str):
    """Tests the Program.concat method.

    Validates that concatenating two Programs produces the expected sequence of instructions and loops in order.
    """
    assert str(program_1.concat(program_2)) == expected_result


@pytest.mark.parametrize("num_repetitions, content, expected_result", [
    (5, 'a a a', '15a'),
    (3, 'a a b', '3*[2a,b]'),
    (2, 'a b a b a b', '6*[a,b]')
])
def test_loop_fold(num_repetitions: int, content: str, expected_result: str):
    """Tests the Loop.fold method.

    Ensures that repeating instruction sequences are folded into Loop objects correctly, and that nested repetitions are
     detected and folded recursively.
    """
    assert str(Loop.fold(num_repetitions, content.split())) == expected_result
