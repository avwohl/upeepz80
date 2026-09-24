"""Relative jumps are made only where they reach, counting bytes."""

from upeepz80 import optimize

from tests._equiv import assert_equivalent, instrs


def _pad(n: int) -> str:
    """``n`` bytes (a multiple of 3) of code no rewrite shortens."""
    return "".join(f"\tld hl,(V+{i})\n" for i in range(n // 3))


def test_relative_jump_measured_in_bytes():
    # 126 bytes between the jump and its target: in range.
    out = optimize("\tjp z,L1\n" + _pad(126) + "L1:\n\tret\n")
    assert instrs(out)[0] == "jr z,L1"
    # 129: not, although it is only 43 lines.
    out = optimize("\tjp z,L1\n" + _pad(129) + "L1:\n\tret\n")
    assert instrs(out)[0] == "jp z,L1"
    # Backwards: a jr at the end of 126 bytes reaches -128.
    out = optimize("L1:\n" + _pad(126) + "\tjp nz,L1\n\tret\n")
    assert instrs(out)[-2] == "jr nz,L1"
    out = optimize("L1:\n" + _pad(129) + "\tjp nz,L1\n\tret\n")
    assert instrs(out)[-2] == "jp nz,L1"


def test_relative_jump_not_across_what_cannot_be_measured():
    out = optimize("\tjp L1\n\tds N\nL1:\n\tret\n")
    assert instrs(out)[0] == "jp L1"
    out = optimize("\tjp L1\n\tds 200\nL1:\n\tret\n")
    assert instrs(out)[0] == "jp L1"
    out = optimize("\tjp L1\n\tdb 'hello'\nL1:\n\tret\n")
    assert instrs(out)[0] == "jr L1"


def test_djnz_reaches_its_target():
    out = optimize("L1:\n" + _pad(126) + "\tdec b\n\tjp nz,L1\n\tcp b\n\tret\n")
    assert "djnz L1" in instrs(out)
    out = optimize("L1:\n" + _pad(129) + "\tdec b\n\tjp nz,L1\n\tcp b\n\tret\n")
    assert "djnz L1" not in instrs(out)


def test_threaded_jr_stays_in_range():
    """Jump threading gives a jr its final target only if it reaches it."""
    src = "\tjr L1\n" + _pad(60) + "L1:\n\tjp L2\n" + _pad(129) + "L2:\n\tret\n"
    out = optimize(src)
    assert instrs(out)[0] == "jr L1", out
    src = "\tjr L1\n" + _pad(30) + "L1:\n\tjp L2\n" + _pad(30) + "L2:\n\tret\n"
    assert instrs(optimize(src))[0] == "jr L2"


def test_mul_strength_only_where_it_does_not_grow():
    kill = "\n\tld a,1\n\tld bc,0\n\tld de,0\n\tcp b\n\tret\n"
    assert instrs(optimize("\tld de,64\n\tcall ??mul16" + kill)).count("add hl,hl") == 6
    assert "call ??mul16" in instrs(optimize("\tld de,128\n\tcall ??mul16" + kill))


def test_relative_jump_keeps_the_label_on_its_line():
    """`L4: jp L3' made relative was written `jr L3', and jumps to L4 went
    nowhere."""
    assert_equivalent("\tjp X\nL1:\tjp L2\nL4:\tjp L3\nX:\n\tjp L4\nL2:\n\tld a,1\n\tret\n"
                      "L3:\n\tld a,2\n\tret\n")
