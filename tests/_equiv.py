"""What the regression tests share: run code before and after optimization.

A case is run on tests/z80sim.py from many random states, and the original and
the optimized code must end the same: every register, the flags and memory.
A case ends in ``ret`` or ``jp 0``, past which the optimizer has to take every
register and flag for live.
"""

import random
import re
from collections.abc import Iterable

from upeepz80 import optimize
from upeepz80.z80 import strip_comment

from tests.peepfuzz import invalid_instructions
from tests.z80sim import CODE_BASE, FLAG_MASK, Machine

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


def state(m: Machine, ignore: Iterable[int] = range(0)) -> dict:
    mem = bytearray(m.mem[:0xE000])
    for a in ignore:
        mem[a] = 0
    return dict(m.r, f=m.f & FLAG_MASK, ix=m.ix, sp=m.sp, mem=bytes(mem))


def patched(asm: str, patch: dict[str, str]) -> tuple[str, list[str]]:
    """``asm`` with the instruction after each label of ``patch`` replaced
    by the instructions it maps the label to (``"nop\\n\\tnop"`` for two);
    and the instructions replaced, one blank between opcode and operands."""
    lines = asm.split("\n")
    was = []
    for label, becomes in patch.items():
        at = next(i for i, line in enumerate(lines)
                  if re.match(re.escape(label) + r"::?(\s|;|$)", line))
        name, text = re.match(r"([^:]*::?)(.*)$", lines[at]).groups()
        if strip_comment(text).strip():  # `SW: jp SKIP'
            lines[at] = name + "\n\t" + becomes
        else:  # `SW:' alone on its line
            at += 1
            while not strip_comment(lines[at]).strip():
                at += 1
            text = lines[at]
            assert text[:1].isspace(), f"no instruction after {label}: {text!r}"
            lines[at] = "\t" + becomes
        was.append(" ".join(strip_comment(text).split()))
    return "\n".join(lines), was


def assert_equivalent(src: str, states: int = 40, entry: str = "", other: str = "",
                      ignore: Iterable[int] = range(0), patch: dict[str, str] | None = None
                      ) -> str:
    """Optimize ``src``; the result must be valid and do what ``src`` does.

    ``entry`` is code run first, outside what is optimized: another module's
    way in, such as ``jp NAME`` to a label ``src`` exports.  ``other`` is
    code after it, outside what is optimized too: another module's routines,
    which ``src`` names as ``extrn``.  The bytes at the addresses ``ignore``
    are not compared: data the optimizer may change, such as a table of
    addresses of code.

    ``patch`` is what the program writes over instructions of ``src`` before
    it runs them: for a label, the instructions the bytes it writes over the
    one after the label make of it (``nop`` for a ``ret`` written over with
    0).  z80sim does not put the bytes of code in memory, so ``src`` and the
    optimized code are both run as the program makes them: with that
    instruction replaced, which the optimizer must leave as it is.  The
    bytes of code, which the program writes and the optimizer moves, are not
    compared."""
    out = optimize(src)
    assert not invalid_instructions(out), out
    old, new = src, out
    if patch:
        old, was = patched(src, patch)
        new, now = patched(out, patch)
        assert now == was, f"the instructions the program writes over were changed\n{out}"
        ignore = [*ignore, *range(CODE_BASE, CODE_BASE + 0x1000)]
    rng = random.Random(1)
    for _ in range(states):
        seed = rng.randrange(1 << 30)
        before = state(run(entry + old + other, random.Random(seed)), ignore)
        after = state(run(entry + new + other, random.Random(seed)), ignore)
        diff = {k: (before[k], after[k]) for k in before if before[k] != after[k] and k != "mem"}
        if before["mem"] != after["mem"]:
            diff["mem"] = [hex(i) for i in range(0xE000) if before["mem"][i] != after["mem"][i]][:8]
        assert not diff, f"{diff}\n--- source\n{src}\n--- optimized\n{out}"
    return out


def instrs(asm: str) -> list[str]:
    """The instructions of ``asm``, one blank between opcode and operands."""
    return [" ".join(l.split()) for l in asm.split("\n") if l.strip() and not l.strip().endswith(":")]
