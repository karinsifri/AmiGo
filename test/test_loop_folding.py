import pytest

from src.instructions.loop_folding import fold_loops, Program


@pytest.mark.parametrize("instructions, folded_instructions", [
    ('a', 'a'),
    ('a a', '2a'),
    ('a a b a a b', '2*[2a,b]'),
    ('a b a b a b a b', '4*[a,b]'),
])
def test_loop_folding(instructions, folded_instructions):
    """ Test the fold_instructions function """
    actual = Program.fold_instructions(instructions.split())
    assert str(actual) == folded_instructions
