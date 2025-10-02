import pytest

from src.instructions.loop_folding import Program, Loop, find_loop


@pytest.mark.parametrize("instructions, folded_instructions", [
    ('a', 'a'),
    ('a a', '2a'),
    ('a a b a a b', '2*[2a,b]'),
    ('a b a b a b a b', '4*[a,b]'),
])
def test_loop_folding(instructions: str, folded_instructions: str):
    """ Test the fold_instructions function """
    actual = Program.fold_instructions(instructions.split())
    assert str(actual) == folded_instructions


@pytest.mark.parametrize("pattern, tail, expected_loop, expected_tail", [
    ('a', 'b', 'None', 'a b'),
    ('a', 'a b', '2a', 'b'),
    ('a b c', 'a b c a b c a b a c', '3*[a,b,c]', 'a b a c'),
    ('a b', 'a b a b a b a b a b', '6*[a,b]', ''),
    ('a b', 'a b a b a c b a b a b', '3*[a,b]', 'a c b a b a b'),
    ('a b', 'c a b a b a b a b a b', 'None', 'a b c a b a b a b a b a b'),
])
def test_find_loop(pattern: str, tail: str, expected_loop: str, expected_tail: str):
    returned_loop, returned_tail = find_loop(pattern.split(), tail.split())
    assert str(returned_loop) == expected_loop and " ".join(returned_tail) == expected_tail


@pytest.mark.parametrize("loop, rest, expected_result", [
    (Loop(3, Program(['a'])), Program([Loop(2, Program(['a'])), 'b', 'a']), '5a,b,a'),
    (Loop(2, Program(['a', 'b'])), Program(['a', 'b', 'c']), '3*[a,b],c'),
    (Loop(2, Program(['a', 'b'])), Program(['c', 'a', 'b']), '2*[a,b],c,a,b'),
    (Loop(5, Program(['a'])), Program([]), '5a'),
])
def test_loop_extend(loop: Loop, rest: Program, expected_result: str):
    assert str(loop.extend(rest)) == expected_result


@pytest.mark.parametrize("program_1, program_2, expected_result", [
    (Program(['a']), Program(['b', 'a']), 'a,b,a'),
    (Program(['a', 'b', Loop(4, Program(['c', 'd', 'e'])), 'f', 'g', 'h']), Program(['a', Loop(3, Program(['b']))]),
     'a,b,4*[c,d,e],f,g,h,a,3b'),
    (Program(['h', 'e', 'l', 'l', 'o']), Program([]), 'h,e,l,l,o')
])
def test_program_extend(program_1: Program, program_2: Program, expected_result: str):
    assert str(program_1.extend(program_2)) == expected_result


@pytest.mark.parametrize("num_repetitions, content, expected_result", [
    (5, 'a a a', '15a'),
    (3, 'a a b', '3*[2a,b]'),
    (2, 'a b a b a b', '6*[a,b]'),
])
def test_loop_fold(num_repetitions: int, content: str, expected_result: str):
    assert str(Loop.fold(num_repetitions, content.split())) == expected_result
