"""Every rewrite writes only instructions the Z80 has."""

import pytest

from upeepz80 import PeepholeOptimizer, optimize
from upeepz80.z80 import UNKNOWN, effect

from tests._equiv import assert_equivalent, instrs


def test_ld_hl_const_byte_is_the_low_byte():
    """`ld hl,299 / ld a,l' is not `ld a,299' - which does not assemble, and
    here L was still needed as well (release difftest seed 1063)."""
    src = "\tld\thl,299\n\tld\ta,l\n\tld\tl,a\n\tld\th,0\n\tld\t(W),hl\n\tld\thl,5\n\tjp\t0\n"
    assert_equivalent(src)
    out = assert_equivalent("\tld hl,299\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[0] == "ld a,43"


def test_ld_hl_of_an_address_is_not_a_byte():
    """`ld hl,NAME / ld a,l' is left as it is: a byte of a relocatable or
    external address is up to the assembler and linker (um80 and ul80 take
    it, others need not).  A name the text sets with `equ' is a constant."""
    out = optimize("\tld hl,W\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert "ld a,W" not in instrs(out)
    out = optimize("K\tequ\t300\n\tld hl,K\n\tld e,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[1] == "ld e,44"


@pytest.mark.parametrize("op", ["inc", "dec"])
def test_inc_in_memory_through_ix_is_inc_ix(op):
    """A REENTRANT procedure's BYTE loop: `ld a,(ix+n) / inc a / ld (ix+n),a'
    became `ld hl,ix+n / inc (hl)', which is not a Z80 instruction."""
    src = f"\tld\ta,(ix+-1)\n\t{op}\ta\n\tld\t(ix+-1),a\n\tld\ta,(ix+4)\n\tret\n"
    out = assert_equivalent(src)
    assert instrs(out)[0] == f"{op} (ix+-1)"
    # A read afterwards: unchanged.
    out = assert_equivalent(f"\tld\ta,(ix+3)\n\t{op}\ta\n\tld\t(ix+3),a\n\tld\t(V),a\n\tret\n")
    assert f"{op} (ix+3)" not in instrs(out)


def test_inc_in_memory_through_hl_is_inc_hl():
    out = assert_equivalent("\tld a,(hl)\n\tinc a\n\tld (hl),a\n\tld a,5\n\tret\n")
    assert instrs(out)[0] == "inc (hl)"


def test_every_output_instruction_is_z80():
    src = "\tld a,(ix+5)\n\tinc a\n\tld (ix+5),a\n\tld hl,299\n\tld a,l\n\tld hl,0\n\tret\n"
    for line in instrs(optimize(src)):
        op, _, operands = line.partition(" ")
        assert effect(op, operands) is not UNKNOWN, line


def test_stats_name_what_fired():
    opt = PeepholeOptimizer()
    opt.optimize("\tld a,(ix+5)\n\tinc a\n\tld (ix+5),a\n\tld a,1\n\tret\n")
    assert opt.stats.get("inc_mem") == 1
