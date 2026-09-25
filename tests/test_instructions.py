"""Every rewrite writes only instructions the Z80 has."""

import pytest

from upeepz80 import PeepholeOptimizer, optimize
from upeepz80.z80 import UNKNOWN, effect

from tests._equiv import assert_equivalent, instrs


def test_ld_hl_const_byte_is_the_low_byte():
    """`ld hl,299 / ld a,l' is `ld a,02Bh', the low byte.  `ld a,299' is out
    of range: um80 takes the low byte without a word, other assemblers
    need not.  And here L was still needed as well (release difftest seed
    1063)."""
    src = "\tld\thl,299\n\tld\ta,l\n\tld\tl,a\n\tld\th,0\n\tld\t(W),hl\n\tld\thl,5\n\tjp\t0\n"
    assert_equivalent(src)
    out = assert_equivalent("\tld hl,299\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[0] == "ld a,02Bh"
    out = assert_equivalent("\tld hl,-1\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[0] == "ld a,0FFh"
    # 0 to 255 stays as it is written.
    out = assert_equivalent("\tld hl,0ah\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[0] == "ld a,0ah"


def test_a_number_under_another_radix():
    """Under `.radix 16', 300 is 300H.  `ld hl,300 / ld a,l' became `ld
    a,44', which that radix reads as 44H; the low byte of 300H is 0."""
    out = optimize("\t.radix 16\n\tld hl,300\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert "ld a,44" not in instrs(out)
    assert instrs(out)[1] == "ld hl,300"
    # With a suffix, a number means the same under any radix; what is written
    # back has one too.
    out = optimize("\t.radix 16\n\tld hl,300h\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[1] == "ld a,000h"
    out = optimize("\t.radix 16\nK\tequ 12CH\n\tld hl,K\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[2] == "ld a,02Ch"
    # 64 is 100 under radix 16: no power of two.
    kill = "\n\tld a,1\n\tld bc,0\n\tld de,0\n\tcp b\n\tret\n"
    out = optimize("\t.radix 16\n\tld de,64\n\tcall ??mul16" + kill)
    assert "call ??mul16" in instrs(out)
    # `ds 10' is 16 bytes, and `ds 70' 112: a jr across it may not reach.
    out = optimize("\t.radix 16\n\tjp L1\n\tds 70\n\tds 10\nL1:\n\tret\n")
    assert instrs(out)[1] == "jp L1"
    # `.radix 10' changes nothing.
    out = optimize("\t.radix 10\n\tld hl,300\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[1] == "ld a,02Ch"


def test_ld_hl_of_an_address_is_not_a_byte():
    """`ld hl,NAME / ld a,l' is left as it is: a byte of a relocatable or
    external address is up to the assembler and linker (um80 and ul80 take
    it, others need not).  A name the text sets with `equ' is a constant."""
    out = optimize("\tld hl,W\n\tld a,l\n\tld hl,0\n\tjp 0\n")
    assert "ld a,W" not in instrs(out)
    out = optimize("K\tequ\t300\n\tld hl,K\n\tld e,l\n\tld hl,0\n\tjp 0\n")
    assert instrs(out)[1] == "ld e,02Ch"


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
