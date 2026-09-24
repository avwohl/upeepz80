"""The differential test (tests/peepfuzz.py), a few hundred programs of it.

Each program is run before and after optimization on the Z80 interpreter
from random states and must end in the same state; see tests/peepfuzz.py,
which runs any number of seeds from the command line.
"""

import pytest

from tests.peepfuzz import check

BLOCKS = [(first, 100) for first in (1, 1001, 2001)]


@pytest.mark.parametrize("first,count", BLOCKS)
def test_optimized_programs_do_what_the_originals_do(first, count):
    failures = [rep for seed in range(first, first + count) if (rep := check(seed))]
    assert not failures, failures[0]
