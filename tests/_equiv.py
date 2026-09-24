"""What the regression tests share: run code before and after optimization.

A case is run on tests/z80sim.py from many random states, and the original and
the optimized code must end the same: every register, the flags and memory.
A case ends in ``ret`` or ``jp 0``, past which the optimizer has to take every
register and flag for live.
"""

import random

from upeepz80 import optimize

from tests.peepfuzz import invalid_instructions
from tests.z80sim import FLAG_MASK, Machine

SYMBOLS = {"@B": 0x8000, "W": 0x8002, "V": 0x8004, "X": 0x8006, "W0": 0x8008,
           "W1": 0x800A, "SB": 0x800C, "W4": 0x800E, "COUNT": 0x8010}


def run(src: str, rng: random.Random) -> Machine:
    m = Machine(src, SYMBOLS)
    for k in "abcdehl":
        m.r[k] = rng.randrange(256)
    m.r["h"], m.r["l"] = 0x80, rng.randrange(0x20)
    m.f = rng.randrange(256)
    m.ix = 0x9080
    m.mem[:] = rng.randbytes(0x10000)
    m.run()
    return m


def state(m: Machine) -> dict:
    return dict(m.r, f=m.f & FLAG_MASK, ix=m.ix, sp=m.sp,
                mem=bytes(m.mem[:0xE000]))


def assert_equivalent(src: str, states: int = 40) -> str:
    """Optimize ``src``; the result must be valid and do what ``src`` does."""
    out = optimize(src)
    assert not invalid_instructions(out), out
    rng = random.Random(1)
    for _ in range(states):
        seed = rng.randrange(1 << 30)
        before = state(run(src, random.Random(seed)))
        after = state(run(out, random.Random(seed)))
        diff = {k: (before[k], after[k]) for k in before if before[k] != after[k] and k != "mem"}
        if before["mem"] != after["mem"]:
            diff["mem"] = [hex(i) for i in range(0xE000) if before["mem"][i] != after["mem"][i]][:8]
        assert not diff, f"{diff}\n--- source\n{src}\n--- optimized\n{out}"
    return out


def instrs(asm: str) -> list[str]:
    """The instructions of ``asm``, one blank between opcode and operands."""
    return [" ".join(l.split()) for l in asm.split("\n") if l.strip() and not l.strip().endswith(":")]
