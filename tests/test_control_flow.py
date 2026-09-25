"""Tail calls, labels and jump threading keep every way control can go;
and the other rewrites that were wrong whatever was live afterwards."""

import pytest

from upeepz80 import optimize

from tests._equiv import assert_equivalent, instrs


@pytest.mark.parametrize("src", [
    # ld hl,0ffffh / ld a,l / or h was made `ld hl,0ffffh / or a', testing A's old value
    "\txor a\n\tld hl,0ffffh\n\tld a,l\n\tor h\n\tjr z,L1\n\tld b,1\nL1:\n\tld hl,0\n\tcp b\n\tret\n",
    # ld l,(hl) twice reads two different bytes
    "\tld hl,V\n\tld l,(hl)\n\tld l,(hl)\n\tld (W),hl\n\tret\n",
    # a conditional call before ret is not a tail call
    "\tor a\n\tcall z,S\n\tret\n\tld a,1\nS:\n\tld a,2\n\tret\n",
    # a label after `ret z' is reached by falling through
    "\tor a\n\tret z\nL1:\n\tjp L2\nX:\n\tld a,1\nL2:\n\tld b,2\n\tret\n",
    # a label's own instruction comes before the next line's jump
    "\tjp L1\n\tld a,1\nL1:\tld a,5\n\tjp L2\n\tld a,2\nL2:\n\tret\n",
    # jump threading gave the jump on a label's line a new target and lost the label
    "\tjp X\nL1:\tjp L2\nX:\n\tjp L1\nL2:\n\tjp L3\n\tld a,1\nL3:\n\tld a,2\n\tret\n",
])
def test_control_and_values_kept(src):
    assert_equivalent(src)


def test_labels_named_in_a_table_are_kept():
    src = "\tjp (hl)\nTAB:\n\tdw L0,L1\n\tjp X\nL1:\n\tjp L2\nL0:\n\tret\nL2:\n\tret\n"
    out = optimize(src)
    assert "L1:" in out


def test_dead_store_read_as_part_of_a_word_is_kept():
    """A sixteen-bit load from the byte before reads the stored byte too."""
    src = "P:\n\tld (??AUTO+3),a\n\tret\nQ:\n\tld hl,(??AUTO+2)\n\tret\n??AUTO:\n\tds 4\n"
    assert "(??AUTO+3),a" in optimize(src)
    src = "P:\n\tld (??AUTO+1),a\n\tret\nQ:\n\tld de,(??AUTO)\n\tret\n??AUTO:\n\tds 4\n"
    assert "(??AUTO+1),a" in optimize(src)
    src = "P:\n\tld (??AUTO+3),a\n\tret\nQ:\n\tld a,(??AUTO+2)\n\tret\n??AUTO:\n\tds 4\n"
    assert "(??AUTO+3),a" not in optimize(src)


# ---- `NAME::' is M80's way to write a PUBLIC label ------------------------------

def test_a_routine_exported_with_double_colon_returns_anywhere():
    """`foo::' exports foo, so another module may call it and read HL after
    it returns.  The routine counted as closed, returning only to the call
    in this module, and `ld hl,5 / ld a,l' became `ld a,5'."""
    src = "\tcall foo\n\tld hl,0\n\tret\nfoo::\n\tld hl,5\n\tld a,l\n\tret\n"
    out = assert_equivalent(src, entry="\tcall foo\n\tjp 0\n")
    assert "ld hl,5" in instrs(out)
    # Spelled `public foo' and `foo:', it was already left alone.
    out = optimize("\tpublic foo\n" + src.replace("foo::", "foo:"))
    assert "ld hl,5" in instrs(out)


def test_a_label_exported_with_double_colon_is_kept():
    """Jump threading removed `bar:: jp L2' as a label nothing names, and a
    module that calls bar no longer linked."""
    src = "\tld a,1\n\tjp L2\nbar::\tjp L2\nL2:\tret\n"
    out = assert_equivalent(src, entry="\tld a,7\n\tcall bar\n\tjp 0\n")
    assert any(line.startswith("bar::") for line in out.split("\n")), out
    # Alone on its line, with the jump on the next.
    src = "\tld a,1\n\tjp L2\nbar::\n\tjp L2\nL2:\tret\n"
    out = assert_equivalent(src, entry="\tld a,7\n\tcall bar\n\tjp 0\n")
    assert "bar::" in out.split("\n"), out


def test_no_tail_call_to_a_routine_that_looks_under_its_return_address():
    """`call P / ret' is `jp P' only if P finds the same thing on the stack
    either way.  This P takes two return addresses off and puts one back:
    called, it returns to its caller's caller; jumped to, one level
    further."""
    src = ("\tld hl,0\n\tcall Q\n\tld hl,1\n\tjp 0\nQ:\n\tcall P\n\tret\n"
           "P:\n\tpop de\n\tpop bc\n\tpush de\n\tld de,0\n\tld bc,0\n\tret\n")
    out = assert_equivalent(src)
    assert "call P" in instrs(out)
    # And a routine that calls one: its stack after the call is not known.
    src = ("\tld hl,0\n\tcall S\n\tld hl,1\n\tjp 0\nS:\n\tcall Q\n\tret\nQ:\n\tcall P\n\tld a,1\n\tret\n"
           "P:\n\tpop de\n\tpop bc\n\tpush de\n\tld de,0\n\tld bc,0\n\tret\n")
    out = assert_equivalent(src)
    assert "call Q" in instrs(out)


# THERE reads the zero flag: it is not where P's `ret' was taken to go.
THERE = "THERE:\n\tld b,0\n\tjr nz,L1\n\tld b,1\nL1:\n\tjp 0\n"


@pytest.mark.parametrize("src", [
    # P writes THERE over its return address, through a pointer from SP.
    ("\tcall P\n\tcp b\n\tjp 0\nP:\n\tld hl,0\n\tadd hl,sp\n\tld de,THERE\n"
     "\tld (hl),e\n\tinc hl\n\tld (hl),d\n\tld a,0\n\tret\n" + THERE),
    # S writes it over its caller's.
    ("\tcall P\n\tcp b\n\tjp 0\nP:\n\tcall S\n\tld a,0\n\tret\nS:\n\tld hl,2\n\tadd hl,sp\n"
     "\tld de,THERE\n\tld (hl),e\n\tinc hl\n\tld (hl),d\n\tret\n" + THERE),
    # R2 writes it through a pointer R1 kept, which is where R2's is.
    ("\tcall R1\n\tcall R2\n\tcp b\n\tjp 0\nR1:\n\tld (W),sp\n\tret\nR2:\n\tld hl,(W)\n"
     "\tld de,THERE\n\tld (hl),e\n\tinc hl\n\tld (hl),d\n\tld a,0\n\tret\n" + THERE),
])
def test_a_return_address_changed_through_a_pointer_made_from_sp(src):
    """P's `ret' was taken to go back to its call, where `cp b' writes
    the flags, and `ld a,0' became `xor a'.  It goes to THERE."""
    out = assert_equivalent(src)
    assert "ld a,0" in instrs(out)


def test_no_tail_call_to_a_routine_that_reads_above_its_return_address():
    """RD reads the word its caller's caller pushed, through a pointer from
    SP.  `call RD / ret' as `jp RD' leaves one return address fewer
    between, and RD reads another word."""
    src = ("\tld hl,7\n\tpush hl\n\tcall QD\n\tpop hl\n\tjp 0\nQD:\n\tcall RD\n\tret\n"
           "RD:\n\tld hl,4\n\tadd hl,sp\n\tld a,(hl)\n\tret\n")
    out = assert_equivalent(src)
    assert "call RD" in instrs(out)


@pytest.mark.parametrize("src", [
    # W points HL where `call RB' puts RB's return address, and RB reads it.
    ("\tcall W\n\tret\nW:\n\tld hl,0\n\tadd hl,sp\n\tdec hl\n\tdec hl\n\tcall RB\n\tret\n"
     "RB:\n\tld a,(hl)\n\tret\n"),
    # W keeps the pointer; RB reads the high byte through it.
    ("\tcall W\n\tret\nW:\n\tld (PTR),sp\n\tcall RB\n\tret\n"
     "RB:\n\tld hl,(PTR)\n\tdec hl\n\tld a,(hl)\n\tret\n\tdseg\nPTR:\tds 2\n"),
])
def test_no_tail_call_to_a_routine_that_reads_through_its_callers_pointer(src):
    """RB makes no pointer from SP, but reads through one its caller made
    before the call, where RB's return address is.  `call RB / ret' as
    `jp RB' puts none there, and RB reads what was below the stack."""
    out = assert_equivalent(src)
    assert "call RB" in instrs(out)


def test_a_routine_that_writes_through_a_pointer_where_none_is_made_from_sp():
    """Where the text makes no pointer from SP, a write through a pointer
    does not reach a return address, and P's `ret' goes back to its call."""
    src = "\tld hl,W\n\tcall P\n\tcp b\n\tjp 0\nP:\n\tld (hl),e\n\tld a,0\n\tret\n"
    out = assert_equivalent(src)
    assert "xor a" in instrs(out)


def test_dead_store_read_spelled_otherwise_is_kept():
    """A load of the stored byte is a read however its address is written."""
    tail = "\tret\nP:\n\tld (BUF+13),a\n\tret\nBUF:\n\tds 16\n"
    for load in ("ld a,(BUF+0DH)", "ld a,(BUF + 13)", "ld hl,(BUF+12)", "ld de,(BUF+0CH)",
                 "ld a,(BUF+14-1)"):
        assert "(BUF+13),a" in optimize(f"\tcall P\n\t{load}\n" + tail), load
    for load in ("ld a,(BUF+12)", "ld hl,(BUF+14)", "ld a,(BUF)", "ld (BUF+13),hl"):
        assert "(BUF+13),a" not in optimize(f"\tcall P\n\t{load}\n" + tail), load
