import pytest

from src.instructions.loop_folding import fold_loops


@pytest.mark.parametrize("instructions, folded_instructions", [
    ('a', 'a'),
    ('a a', '2a'),
    ('a a b a a b', '2*[2a,b]')
])
def test_loop_folding(instructions, folded_instructions):
    """ Test the fold_loops function """
    assert fold_loops(instructions) == folded_instructions
