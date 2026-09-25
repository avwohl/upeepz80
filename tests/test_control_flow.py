"""Tail calls, labels and jump threading keep every way control can go;
and the other rewrites that were wrong whatever was live afterwards."""

import pytest

from upeepz80 import optimize

from tests._equiv import assert_equivalent, instrs
from tests.z80sim import Machine


def test_the_interpreter_jumps_to_a_name_an_equate_sets_to_code():
    """`ALIAS equ RTN' names RTN's address: a jump or call to ALIAS goes
    there, as it does on an assembler."""
    src = "\tcall ALIAS\n\tjp ALIAS\nALIAS\tequ RTN\nRTN:\n\tinc a\n\tret\n"
    m = Machine(src)
    m.r["a"] = 1
    assert m.run() == "ret"
    assert m.r["a"] == 3


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


# RTN reads the word pushed before QQ was called, through a pointer it makes
# from SP, as RD above does.
RD_MAIN = "\tld de,5A5Ah\n\tpush de\n\tcall QQ\n\tpop de\n\tld hl,0\n\tjp 0\n"
RD_RTN = "RTN:\n\tld ix,9080h\n\tld hl,4\n\tadd hl,sp\n\tld a,(hl)\n\tld hl,0\n\tret\n"


@pytest.mark.parametrize("way", [
    # The label spelled in lower case: M80 does not tell case.
    "QQ:\n\tcall rtn\n\tret\n",
    # A name an equate sets to it.
    "QQ:\n\tcall ALIAS\n\tret\nALIAS\tequ RTN\n",
    "QQ:\n\tcall DSP\n\tret\nDSP:\n\tjp ALIAS\nALIAS\tequ RTN\n",
    # Its address in a register, and a jump to it.
    "QQ:\n\tld hl,RTN\n\tcall DSP\n\tret\nDSP:\n\tjp (hl)\n",
    "QQ:\n\tld ix,RTN\n\tcall DSP\n\tret\nDSP:\n\tjp (ix)\n",
    "QQ:\n\tld iy,RTN\n\tcall DSP\n\tret\nDSP:\n\tjp (iy)\n",
    "QQ:\n\tld hl,RTN\n\tcall DSP\n\tret\nDSP:\n\tpush hl\n\tret\n",
    # ... from a routine DSP calls.
    "QQ:\n\tld hl,RTN\n\tcall DSP\n\tret\nDSP:\n\tcall DSP2\n\tret\nDSP2:\n\tjp (hl)\n",
])
def test_no_tail_call_to_a_routine_reached_where_it_cannot_be_followed(way):
    """QQ reaches RTN other than through a label as it is written, so
    `call RTN' is not seen.  `call ... / ret' as `jp ...' leaves one return
    address fewer between, and RTN reads another word."""
    out = assert_equivalent(RD_MAIN + way + RD_RTN)
    assert any(line.startswith("call ") for line in instrs(out)), out


@pytest.mark.parametrize("src", [
    # WW points HL where `call DSP' puts DSP's return address, and RTN,
    # which DSP goes to through IX, reads it.
    ("\tcall WW\n\tjp 0\nWW:\n\tld hl,0\n\tadd hl,sp\n\tdec hl\n\tdec hl\n\tdec hl\n\tdec hl\n"
     "\tld ix,RTN\n\tcall DSP\n\tret\nDSP:\n\tjp (ix)\nRTN:\n\tld a,(hl)\n\tld ix,9080h\n"
     "\tld hl,0\n\tret\n"),
    # RTN takes its return address and what QQ pushed, and puts the first
    # back: jumped to, DSP would take QQ's return address for the second.
    ("\tld bc,1234h\n\tcall QQ\n\tld de,0\n\tjp 0\nQQ:\n\tpush bc\n\tld hl,RTN\n\tcall DSP\n"
     "\tret\nDSP:\n\tjp (hl)\nRTN:\n\tpop hl\n\tpop de\n\tpush hl\n\tld hl,0\n\tret\n"),
])
def test_no_tail_call_through_jp_to_a_register(src):
    out = assert_equivalent(src)
    assert "call DSP" in instrs(out), out


@pytest.mark.parametrize("body", [
    "\tjp (hl)\n", "\tld a,1\n\thalt\n", "\treti\n", "\tjp RTN+3\n", "\tjp $+3\n",
    "\tjr $+2\n", "\tjp nz,RTN+3\n", "\tld a,1\n\tdb 0\n", "\tld a,1\n\tif 1\n\tret\n\tendif\n",
    "\tmvi a,1\n\tret\n", "\tcall rtn\n\tret\n",
])
def test_no_tail_call_to_a_routine_that_goes_where_it_cannot_be_followed(body):
    """What DSP runs next is not known, and may read above its return
    address."""
    src = "\tcall DSP\n\tret\nDSP:\n" + body + "RTN:\n\tld a,2\n\tret\n"
    assert "call DSP" in instrs(optimize(src))


@pytest.mark.parametrize("src", [
    # a routine of another module, and a number
    "\textrn EXT\n\tcall EXT\n\tret\n",
    "\tcall 5\n\tret\n",
    # a name an equate sets to a number is one
    "BDOS\tequ 5\n\tcall BDOS\n\tret\n",
    "\tcall DSP\n\tret\nDSP:\n\tld c,2\n\tjp BDOS\nBDOS\tequ 5\n",
])
def test_a_tail_call_out_of_the_text_is_made(src):
    """Code that another module holds, or at an address that is a number,
    is taken not to read above its return address (see Known issues)."""
    assert not any(line.startswith("call ") for line in instrs(optimize(src)))


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
