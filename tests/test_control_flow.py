"""Tail calls, labels and jump threading keep every way control can go;
and the other rewrites that were wrong whatever was live afterwards."""

import time

import pytest

from upeepz80 import optimize

from tests._equiv import assert_equivalent, instrs
from tests.z80sim import DATA_BASE, Machine


def test_the_interpreter_jumps_to_a_name_an_equate_sets_to_code():
    """`ALIAS equ RTN' names RTN's address: a jump or call to ALIAS goes
    there, as it does on an assembler."""
    src = "\tcall ALIAS\n\tjp ALIAS\nALIAS\tequ RTN\nRTN:\n\tinc a\n\tret\n"
    m = Machine(src)
    m.r["a"] = 1
    assert m.run() == "ret"
    assert m.r["a"] == 3


def test_the_interpreter_takes_dollar_in_an_equate_for_the_next_instruction():
    """`TABLE equ $' names the address of the instruction after it, as on
    an assembler: TABLE+2 is the `inc a' after `jr X'."""
    m = Machine("\tjp TABLE+2\nTABLE\tequ $\n\tjr X\n\tinc a\nX:\n\tinc a\n\tret\n")
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


@pytest.mark.parametrize("src", [
    # RTN, which DSP goes on to through `jp (hl)', writes THERE over QQ's
    # return address through a pointer from SP.
    ("\tcall QQ\n\tcp b\n\tjp 0\nQQ:\n\tld hl,RTN\n\tcall DSP\n\tld a,0\n\tret\nDSP:\n\tjp (hl)\n"
     "RTN:\n\tld hl,2\n\tadd hl,sp\n\tld de,THERE\n\tld (hl),e\n\tinc hl\n\tld (hl),d\n\tret\n" + THERE),
    # RTN takes QQ's return address off the stack and puts THERE instead.
    ("\tcall QQ\n\tcp b\n\tjp 0\nQQ:\n\tld hl,RTN\n\tcall DSP\n\tld a,0\n\tret\nDSP:\n\tjp (hl)\n"
     "RTN:\n\tpop de\n\tpop bc\n\tld hl,THERE\n\tpush hl\n\tpush de\n\tld bc,0\n\tld de,0\n\tret\n"
     + THERE),
    ("\tcall QQ\n\tcp b\n\tjp 0\nQQ:\n\tld hl,RTN\n\tcall DSP\n\tld a,0\n\tret\nDSP:\n\tpush hl\n"
     "\tret\nRTN:\n\tpop de\n\tpop bc\n\tld hl,THERE\n\tpush hl\n\tpush de\n\tld bc,0\n\tld de,0\n"
     "\tret\n" + THERE),
])
def test_a_return_address_changed_by_code_reached_where_it_cannot_be_followed(src):
    """QQ's `ret' was taken to go back to its call, where `cp b' writes the
    flags, and `ld a,0' became `xor a'.  DSP goes on to RTN, which QQ's
    `ret' sends to THERE."""
    out = assert_equivalent(src)
    assert "ld a,0" in instrs(out)


# SWAP puts THERE in place of its caller's return address: it takes its own
# off the stack, and its caller's, and puts its own back.
SWAP = ("SWAP:\n\tpop hl\n\tpop de\n\tld de,THERE\n\tpush de\n\tpush hl\n\tret\n"
        "THERE:\n\tld a,1\n\tret\n")
SWAP_MAIN = "START:\n\tcall QQ\n\tld hl,0\n\tld de,0\n\tjp 0\n"


@pytest.mark.parametrize("caller", [
    "CALLER:\n\tcall SWAP\n\tjp EXT\n",
    "CALLER:\n\tcall MID\n\tjp EXT\nMID:\n\tjp SWAP\n",
    "CALLER:\n\tcall SWAP\n\tld b,3\nLOOP:\n\tdjnz LOOP\n\tjp EXT\n",
    # PEEK only reads the word, and stores it.
    "CALLER:\n\tcall PEEK\n\tjp EXT\nPEEK:\n\tpop hl\n\tpop de\n\tpush de\n\tpush hl\n"
    "\tld (W),de\n\tret\n",
])
def test_no_tail_call_to_a_routine_whose_callee_changes_its_return_address(caller):
    """CALLER leaves by a jump, not a `ret' at a height not known, so it
    was not taken to come back irregularly, nor to read above its return
    address; but SWAP changes the word above its own, CALLER's.  With `call CALLER', that is the
    `ret' after it in QQ, which THERE's `ret' then runs; with `jp CALLER',
    QQ's own return address, and THERE returns to what is above it."""
    src = SWAP_MAIN + "QQ:\n\tcall CALLER\n\tret\n" + caller + SWAP
    out = assert_equivalent(src, entry="\tjp START\nEXT:\n\tret\n")
    assert "call CALLER" in instrs(out)


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


def test_no_tail_call_where_the_caller_may_have_pushed():
    """`push bc / call SHOWP / ret' as `push bc / jp SHOWP' leaves BC where
    SHOWP's return address was.  SHOWP, of another module, takes its
    argument off the stack: jumped to, it takes QQ's return address for
    the argument, and returns to BC."""
    entry = "\tcall QQ\n\tld de,0\n\tjp 0\nSHOWP:\n\tpop hl\n\tpop de\n\tpush hl\n\tld hl,0\n\tret\n"
    src = "\textrn SHOWP\nQQ:\n\tld bc,1234h\n\tpush bc\n\tcall SHOWP\n\tret\n"
    out = assert_equivalent(src, entry=entry)
    assert "call SHOWP" in instrs(out)
    # Nor where how much is on the stack is not known.
    assert "call EXT" in instrs(optimize("\textrn EXT\n\tld sp,hl\n\tcall EXT\n\tret\n"))
    # Code that nothing reaches is not run.
    assert "call EXT" not in instrs(optimize("\textrn EXT\n\tjp 0\nP:\n\tcall EXT\n\tret\n"))


# SHOWP, of another module, takes its argument off the stack, as SHOWP above.
SHOWP = ("\tjp START\nSHOWP:\n\tpop hl\n\tpop de\n\tpush hl\n\tld a,e\n\tld (@B),a\n\tret\n"
         "SHOW0:\n\tld a,c\n\tld (@B),a\n\tret\n")
SHOWP_MAIN = "START:\n\tld bc,5A5Ah\n\tcall QQ\n\tld hl,0\n\tld de,0\n\tjp 0\n"


@pytest.mark.parametrize("way", [
    "\tld hl,HND\n\tjp (hl)\n",
    "\tld ix,HND\n\tjp (ix)\n",
    "\tld hl,(TBL)\n\tjp (hl)\nTBL:\tdw HND\n",
    "\tld hl,HND\n\tpush hl\n\tret\n",
    "\tjp ALIAS\nALIAS\tequ HND\n",
    "\tjp HND\n",  # followed: HND's height is 1, as 0.2.5 had it
])
def test_no_tail_call_where_a_jump_not_followed_leaves_an_argument_pushed(way):
    """QQ pushes SHOWP's argument, and goes on to HND where the optimizer
    does not follow it.  HND's height, counted from its label, is 0, but
    what QQ pushed is on the stack: `call SHOWP / ret' as `jp SHOWP' would
    take QQ's return address for the argument."""
    src = SHOWP_MAIN + "QQ:\n\tpush bc\n" + way + "HND:\n\tcall SHOWP\n\tret\n"
    out = assert_equivalent(src, entry=SHOWP)
    assert "call SHOWP" in instrs(out), out


def test_no_tail_call_after_a_jump_past_itself_with_an_argument_pushed():
    """`jp $+3' goes to the line after it, which is not followed."""
    out = optimize("\textrn SHOWP\nQQ::\n\tpush bc\n\tjp $+3\n\tcall SHOWP\n\tret\n")
    assert "call SHOWP" in instrs(out), out


@pytest.mark.parametrize("way", [
    "\tld hl,HND\n\tjp (hl)\n",
    "\tld hl,HND\n\tpush hl\n\tret\n",
])
def test_a_tail_call_where_a_jump_not_followed_leaves_nothing_pushed(way):
    """Where every such jump is made with nothing pushed, the height from
    HND's label is exact."""
    src = SHOWP_MAIN + "QQ:\n" + way + "HND:\n\tcall SHOW0\n\tret\n"
    out = assert_equivalent(src, entry=SHOWP)
    assert "jp SHOW0" in instrs(out), out


def test_a_routine_that_writes_through_a_pointer_where_none_is_made_from_sp():
    """Where the text makes no pointer from SP, a write through a pointer
    does not reach a return address, and P's `ret' goes back to its call."""
    src = "\tld hl,W\n\tcall P\n\tcp b\n\tjp 0\nP:\n\tld (hl),e\n\tld a,0\n\tret\n"
    out = assert_equivalent(src)
    assert "xor a" in instrs(out)


# ---- PL/M-80's convention: the callee removes the arguments pushed for it -----
#
# Three or more arguments: all but the last two are pushed, the next-to-last is
# in BC and the last in DE, and the callee takes the pushed ones off the stack
# before it returns.  `call P / ret' is then `jp P' only where nothing has been
# pushed since the routine was entered: jumped to, P would take the return
# address for its first argument.

# P3(a, b, c): c in DE, b in BC, a pushed; removed at entry.
_P3 = ("P3:\n\tld (W4),de\n\tld (W1),bc\n\tpop hl\n\tex (sp),hl\n\tld (W0),hl\n"
       "\tld hl,(W0)\n\tld de,(W1)\n\tadd hl,de\n\tld de,(W4)\n\tadd hl,de\n\tld (X),hl\n\tret\n")
# P4(a, b, c, d): two pushed, removed at entry.
_P4 = ("P4:\n\tld (COUNT),de\n\tld (SB),bc\n\tpop hl\n\tpop de\n\tpop bc\n\tpush hl\n"
       "\tld (V),de\n\tld (W),bc\n\tret\n")
_DRIVER = "\tcall Q\n\tld (W),hl\n\tret\n"


def test_no_tail_call_with_an_argument_pushed_for_another_modules_callee():
    """P3 is in another module, so nothing here says it removes an argument."""
    src = ("\textrn P3\nQ::\n\tld hl,1111h\n\tpush hl\n\tld bc,2222h\n\tld de,3333h\n"
           "\tcall P3\n\tret\n")
    out = assert_equivalent(src, entry=_DRIVER, other=_P3)
    assert "call P3" in instrs(out)


def test_no_tail_call_where_the_stack_is_back_at_zero_by_a_callees_count_only():
    """PX's own call of P4 pushes two words that P4 removes, which PX counts as
    still there; its last argument it takes off only at its exit.  Its
    `call PX / ret' in Q has an argument pushed under the return address."""
    src = ("\textrn P4\nQ::\n\tld hl,1111h\n\tpush hl\n\tld bc,2222h\n\tld de,3333h\n"
           "\tcall PX\n\tret\n"
           "PX:\n\tld (W4),de\n\tld (W1),bc\n\tld hl,5\n\tpush hl\n\tpush hl\n\tld bc,6\n"
           "\tld de,7\n\tcall P4\n\tpop hl\n\tpop de\n\tld (W0),de\n\tpush hl\n\tret\n")
    out = assert_equivalent(src, entry=_DRIVER, other=_P4)
    assert "call PX" in instrs(out)


def test_no_tail_call_through_an_address_with_an_argument_pushed():
    """A CALL through an address: the helper jumps to what HL holds."""
    src = ("Q::\n\tld hl,1111h\n\tpush hl\n\tld bc,2222h\n\tld de,3333h\n\tld hl,P3\n"
           "\tcall ??jphl\n\tret\n??jphl:\n\tjp (hl)\n" + _P3)
    out = assert_equivalent(src, entry=_DRIVER)
    assert "call ??jphl" in instrs(out)


def test_a_tail_call_with_every_argument_in_a_register_is_made():
    """Two arguments are both in registers: nothing is pushed for the callee."""
    src = "\textrn P2\nQ::\n\tld bc,2222h\n\tld de,3333h\n\tcall P2\n\tret\n"
    p2 = "P2:\n\tld (W1),bc\n\tld (W4),de\n\tld hl,(W1)\n\tadd hl,de\n\tret\n"
    out = assert_equivalent(src, entry=_DRIVER, other=p2)
    assert "jp P2" in instrs(out)


def test_dead_store_read_spelled_otherwise_is_kept():
    """A load of the stored byte is a read however its address is written."""
    tail = "\tret\nP:\n\tld (BUF+13),a\n\tret\nBUF:\n\tds 16\n"
    for load in ("ld a,(BUF+0DH)", "ld a,(BUF + 13)", "ld hl,(BUF+12)", "ld de,(BUF+0CH)",
                 "ld a,(BUF+14-1)"):
        assert "(BUF+13),a" in optimize(f"\tcall P\n\t{load}\n" + tail), load
    for load in ("ld a,(BUF+12)", "ld hl,(BUF+14)", "ld a,(BUF)", "ld (BUF+13),hl"):
        assert "(BUF+13),a" not in optimize(f"\tcall P\n\t{load}\n" + tail), load


# ---- Where the return address is ---------------------------------------------
#
# A `ret' goes back to the call only where the return address its routine was
# entered with is at the top of the stack.  Where it may not be, it goes where
# the optimizer cannot follow, and the code there may be entered with what the
# routine's caller pushed still on the stack.

HND = "HND:\n\tcall SHOWP\n\tret\n"


@pytest.mark.parametrize("way", [
    # DSP puts HND in place of its return address.
    "QQ:\n\tpush bc\n\tld hl,HND\n\tcall DSP\n\tld a,1\n\tret\nDSP:\n\tex (sp),hl\n\tret\n",
    # PEEK reads what QQ pushed above its return address, so the height
    # after its call is not known, and QQ's `ret' takes HND.
    "QQ:\n\tpush bc\n\tld hl,HND\n\tpush hl\n\tcall PEEK\n\tret\n"
    "PEEK:\n\tpop de\n\tpop hl\n\tpush hl\n\tpush de\n\tld de,0\n\tret\n",
    # W writes HND over its return address, through a pointer from SP.
    "QQ:\n\tpush bc\n\tcall W\n\tld a,1\n\tret\nW:\n\tld hl,0\n\tadd hl,sp\n\tld de,HND\n"
    "\tld (hl),e\n\tinc hl\n\tld (hl),d\n\tld hl,0\n\tld de,0\n\tret\n",
    # EXT, of another module, returns to HND, which QQ pushed.
    "QQ:\n\tpush bc\n\tld hl,HND\n\tpush hl\n\tjp EXT\n",
])
def test_no_tail_call_where_a_ret_that_may_not_go_back_leaves_an_argument_pushed(way):
    """QQ pushes SHOWP's argument, and a `ret' goes to HND, where the
    optimizer took it to go back to a call.  HND's height, counted from its
    label, is 0, and `call SHOWP / ret' became `jp SHOWP'."""
    out = assert_equivalent(SHOWP_MAIN + way + HND, entry=SHOWP + "EXT:\n\tret\n")
    assert "call SHOWP" in instrs(out), out


def test_no_tail_call_to_a_routine_that_pops_where_the_height_is_not_known():
    """AA pushes BC and goes on into SUBR, which QQ calls: SUBR's height is
    not known, and its `pop hl' takes the return address where QQ calls
    it.  Jumped to, it took QQ's."""
    src = ("START:\n\tcall QQ\n\tld bc,1234h\n\tcall AA\n\tld hl,0\n\tjp 0\n"
           "QQ:\n\tcall SUBR\n\tret\nAA:\n\tpush bc\nSUBR:\n\tpop hl\n\tld (V),hl\n\tjp EXT\n")
    out = assert_equivalent(src, entry="\tjp START\nEXT:\n\tret\n")
    assert "call SUBR" in instrs(out), out


def test_a_tail_call_where_the_return_address_is_on_top():
    """Q has taken its argument off the stack: its return address is on
    top, a word higher than it was entered, and RX, jumped to, returns to
    Q's caller as Q would."""
    src = ("\tld bc,1234h\n\tpush bc\n\tcall Q\n\tld hl,0\n\tjp 0\n"
           "Q:\n\tpop hl\n\tex (sp),hl\n\tld (W),hl\n\tcall RX\n\tret\nRX:\n\tld hl,5\n\tret\n")
    out = assert_equivalent(src)
    assert "call RX" not in instrs(out), out


def test_a_routine_whose_callee_takes_its_return_address_goes_anywhere():
    """P takes QQ's return address for its argument: QQ's `ret' goes to
    THERE, which reads the carry, not back to the call, after which `cp b'
    writes the flags."""
    src = ("\tld hl,THERE\n\tpush hl\n\tcall QQ\n\tcp b\n\tjp 0\nQQ:\n\tscf\n\tld a,0\n"
           "\tcall P\n\tret\nTHERE:\n\tld b,0\n\tjr nc,L1\n\tld b,1\nL1:\n\tjp 0\n"
           "P:\n\tpop hl\n\tex (sp),hl\n\tld (W),hl\n\tld hl,0\n\tret\n")
    out = assert_equivalent(src)
    assert "ld a,0" in instrs(out), out


# ---- Code an address computed from a label reaches ----------------------------

@pytest.mark.parametrize("way", [
    "QQ:\n\tpush bc\n\tld hl,LL+3\n\tjp (hl)\nLL:\n\tjp 0\n\tcall SHOWP\n\tret\n",
    "QQ:\n\tpush bc\n\tjp LL+3\nLL:\n\tjp 0\n\tcall SHOWP\n\tret\n",
    "QQ:\n\tpush bc\n\tld hl,LL+3\n\tpush hl\n\tret\nLL:\n\tjp 0\n\tcall SHOWP\n\tret\n",
    "QQ:\n\tpush bc\n\tjr $+3\n\tret\n\tcall SHOWP\n\tret\n",
    # the second entry of a table of jumps
    "QQ:\n\tpush bc\n\tld hl,TBL+3\n\tjp (hl)\nTBL:\n\tjp E0\n\tjp E1\nE0:\n\tret\nE1:\n"
    "\tcall SHOWP\n\tret\n",
])
def test_no_tail_call_in_code_an_address_computed_from_a_label_reaches(way):
    """QQ pushes SHOWP's argument, and goes on to code that only an
    address computed from a label reaches, which counted as reached by
    nothing, so `call SHOWP / ret' became `jp SHOWP'."""
    out = assert_equivalent(SHOWP_MAIN + way, entry=SHOWP)
    assert "call SHOWP" in instrs(out), out


# QQ goes to entry X of TABLE, computed at run time, with SHOWP's argument
# pushed: H0 takes it off, H1 does not.
JP_TABLE = ("QQ:\n\tpush bc\n\tld a,(X)\n\tand 1\n\tld l,a\n\tld h,0\n\tld d,h\n\tld e,l\n"
            "\tadd hl,hl\n\tadd hl,de\n\tld de,TABLE\n\tadd hl,de\n\tjp (hl)\n")
JP_ENTRIES = "\tjp H0\n\tjp H1\nH0:\n\tpop bc\n\tret\nH1:\n\tcall SHOWP\n\tret\n"


@pytest.mark.parametrize("way", [
    JP_TABLE + "TABLE:\n" + JP_ENTRIES,
    JP_TABLE.replace("ld de,TABLE", "ld de,table") + "TABLE:\n" + JP_ENTRIES,
    JP_TABLE.replace("ld de,TABLE", "ld de,TB") + "TB\tequ TABLE\nTABLE:\n" + JP_ENTRIES,
    JP_TABLE + "TABLE\tequ $\n" + JP_ENTRIES,
    JP_TABLE.replace("ld de,TABLE", "ld de,(TP)") + "TP:\tdw TABLE\nTABLE:\n" + JP_ENTRIES,
    JP_TABLE + "TABLE:\t; the table\n\n\tjp H0\n; the second entry\nT1:\tjp H1\n"
    "H0:\n\tpop bc\n\tret\nH1:\n\tcall SHOWP\n\tret\n",
])
def test_a_table_of_jumps_entered_at_an_offset_computed_at_run_time(way):
    """TABLE's second entry, `jp H1', counted as reached by nothing, and so
    did H1: `call SHOWP / ret' became `jp SHOWP', with QQ's argument still
    pushed.  And `jp H0' became `jr H0', so TABLE+3 was inside `jp H1'."""
    out = assert_equivalent(SHOWP_MAIN + way, entry=SHOWP)
    assert "call SHOWP" in instrs(out), out
    assert "jp H0" in instrs(out) and [x for x in instrs(out) if x.endswith("jp H1")], out


# The same, with entries of four bytes: QQ goes to TABLE+4 or TABLE+8.
JP4_TABLE = ("QQ:\n\tpush bc\n\tld a,(X)\n\tand 1\n\tinc a\n\tld l,a\n\tld h,0\n"
             "\tadd hl,hl\n\tadd hl,hl\n\tld de,TABLE\n\tadd hl,de\n\tjp (hl)\n")
JP4_ENTRIES = ("TABLE:\n\tjp H0\n\t{pad}\n\tjp H1\n\t{pad}\n\tjp H2\n\t{pad}\n"
               "H0:\nH1:\n\tpop bc\n\tret\nH2:\n\tcall SHOWP\n\tret\n")


@pytest.mark.parametrize("pad", ["nop", "db 0", "defb 0", "ds 1", "defs 1"])
def test_a_table_of_jumps_padded_to_four_bytes_keeps_its_size(pad):
    """The entries after the first `nop' counted as reached by nothing:
    `jp H1' became `jr H1', so TABLE+8 was inside `jp H2', and H2's `call
    SHOWP / ret' became `jp SHOWP', with QQ's argument still pushed.
    (z80sim lays data out away from the code, so only `nop' is run.)"""
    src = SHOWP_MAIN + JP4_TABLE + JP4_ENTRIES.format(pad=pad)
    out = assert_equivalent(src, entry=SHOWP) if pad == "nop" else optimize(src)
    assert "call SHOWP" in instrs(out), out
    assert [x for x in instrs(out) if x in ("jp H0", "jp H1", "jp H2")] == \
        ["jp H0", "jp H1", "jp H2"], out


def test_a_table_of_relative_jumps_entered_at_an_offset_computed_at_run_time():
    """`jr H1', at TABLE+2, counted as reached by nothing, and so did H1:
    `call SHOWP / ret' became `jp SHOWP', with QQ's argument still pushed."""
    src = (SHOWP_MAIN + "QQ:\n\tpush bc\n\tld a,(X)\n\tand 1\n\tld l,a\n\tld h,0\n\tadd hl,hl\n"
           "\tld de,TABLE\n\tadd hl,de\n\tjp (hl)\nTABLE:\n\tjr H0\n\tjr H1\n"
           "H0:\n\tpop bc\n\tret\nH1:\n\tcall SHOWP\n\tret\n")
    out = assert_equivalent(src, entry=SHOWP)
    assert "call SHOWP" in instrs(out), out


BIOS_CALLER = ("\tjp START\nSTART:\n\tcall BIOS+3\n\tld (V),a\n\tcall BIOS+6\n\tld (W),a\n"
               "\tcall BIOS\n\tjp 0\n")


@pytest.mark.parametrize("src", [
    "BIOS::\tjp BOOT\n\tjp WBOOT\n\tjp CONST\n",
    "BIOS::\n\tjp BOOT\nWBE:\tjp WBOOT\nCSE:\tjp CONST\n",
    "\tpublic BIOS\nBIOS:\tjp BOOT\n\tjp WBOOT\n\tjp CONST\n",
])
def test_a_vector_of_jumps_another_module_enters_at_an_offset_keeps_its_size(src):
    """Another module calls BIOS+3 and BIOS+6.  The jumps became `jr',
    two bytes each, or all but the first were removed as jumps nothing
    names."""
    src += "BOOT:\n\tld a,1\n\tret\nWBOOT:\n\tld a,2\n\tret\nCONST:\n\tld a,3\n\tret\n"
    out = assert_equivalent(src, entry=BIOS_CALLER)
    assert [line for line in instrs(out) if line.endswith("jp WBOOT")], out


def test_a_long_run_of_named_data_is_looked_at_once():
    """Where the text names a label, the jumps after it, and what pads
    them, are a table.  They were looked for from each such label on, over
    the data after it, which may pad a table, to the end of the run: over
    8,000 labels of data in a row that a table of words names, that was 32
    million lines, and optimize() took 3.9 seconds of CPU time.  A line is
    now looked at once, and it takes 0.14."""
    names = [f"M{k}" for k in range(8000)]
    src = ("\tld hl,TBL0\n\tret\n" +
           "".join(f"TBL{k}:\tdw {','.join(names[k:k + 16])}\n" for k in range(0, 8000, 16)) +
           "".join(f"{name}:\tdb 'x',0\n" for name in names))
    start = time.process_time()
    optimize(src)
    assert time.process_time() - start < 1.0


def test_code_between_a_label_and_an_address_computed_from_it_keeps_its_size():
    """LB+2 is the `jp MM' after `ld a,0'.  `ld a,0' became `xor a', a
    byte, and `jp MM' went, as a jump to the next line: LB+2 was then past
    `cp b'."""
    src = "\tld hl,LB+2\n\tjp (hl)\nLB:\n\tld a,0\n\tjp MM\nMM:\n\tcp b\n\tld hl,0\n\tret\n"
    out = assert_equivalent(src)
    assert instrs(out)[2:4] == ["ld a,0", "jp MM"], out


# ---- A word that holds the address of code ------------------------------------

def test_a_word_of_code_that_is_compared_keeps_its_label():
    """C0 is `jp DONE', and the word TBL holds its address.  It became `dw
    DONE', so the comparison with C0 failed."""
    src = ("\tld hl,(TBL)\n\tld de,C0\n\tor a\n\tsbc hl,de\n\tld a,0\n\tjr nz,X\n\tld a,1\n"
           "X:\n\tld (V),a\n\tld hl,0\n\tld de,0\n\tret\nTBL:\n\tdw C0\nC0:\n\tjp DONE\n"
           "\tld a,2\nDONE:\n\tret\n")
    out = assert_equivalent(src)
    assert "dw C0" in instrs(out), out


def test_a_table_only_jumped_through_is_threaded():
    """The dispatch uplm80 writes for DO CASE: a word of its table is only
    jumped to, with itself in HL, which the code there does not read.  (The
    table itself is not compared.)"""
    src = ("\tld a,(X)\n\tand 1\n\tld l,a\n\tld h,0\n\tadd hl,hl\n\tld de,TBL\n\tadd hl,de\n"
           "\tld e,(hl)\n\tinc hl\n\tld d,(hl)\n\tex de,hl\n\tjp (hl)\nTBL:\n\tdw C0\n\tdw C1\n"
           "C0:\n\tjp DONE\nC1:\n\tld a,1\n\tjp DONE\nDONE:\n\tld hl,0\n\tld de,0\n\tret\n")
    out = assert_equivalent(src, ignore=range(DATA_BASE, DATA_BASE + 4))
    assert "dw DONE" in instrs(out), out
    # Where the code there reads HL, which holds C0 or DONE, it is not.
    src = src.replace("DONE:\n\tld hl,0\n", "DONE:\n\tld (W),hl\n\tld hl,0\n")
    assert "dw DONE" not in instrs(optimize(src))


@pytest.mark.parametrize("read", [
    "\tdec de\n\tld a,(de)\n",
    "\tex de,hl\n\tdec hl\n\tld a,(hl)\n",
])
def test_a_table_whose_target_reads_the_entry_through_de_is_not_threaded(read):
    """The dispatch leaves DE pointing at the entry's high byte, and DONE
    reads the low byte through it: C0's, which it compares with C0, in the
    table as it was, and DONE's once `dw C0' became `dw DONE'."""
    src = ("\tld a,(X)\n\tand 1\n\tld l,a\n\tld h,0\n\tadd hl,hl\n\tld de,TBL\n\tadd hl,de\n"
           "\tld e,(hl)\n\tinc hl\n\tld d,(hl)\n\tex de,hl\n\tjp (hl)\nTBL:\n\tdw C0\n\tdw C1\n"
           "C0:\n\tjp DONE\nC1:\n\tld a,1\n\tjp DONE\nDONE:\n" + read +
           "\tld hl,C0\n\tcp l\n\tld a,0\n\tjr z,EQ\n\tinc a\nEQ:\n\tld (V),a\n"
           "\tld hl,0\n\tld de,0\n\tret\n")
    out = assert_equivalent(src, ignore=range(DATA_BASE, DATA_BASE + 4))
    assert "dw C0" in instrs(out), out


# ---- A jump whose operand an address computed from a label reaches ------------
#
# z80sim does not put the bytes of code in memory, so these look at the code.
# Programs like them, run on um80, ul80 and cpmemu, print something else once
# optimized.

VEC = ("VEC:\n\tjp HANDLER\nHANDLER:\n\tjp REAL\nOTHER:\n\tld a,1\n\tret\nREAL:\n\tld a,2\n\tret\n"
       "MINE:\n\tld a,3\n\tret\n")


@pytest.mark.parametrize("src", [
    # the operand compared with HANDLER
    "\tld hl,(VEC+1)\n\tld de,HANDLER\n\tor a\n\tsbc hl,de\n\tld (W),hl\n\tcall VEC\n\tret\n",
    # kept, and the vector chained to MINE, which goes on to what it held
    "\tld hl,(VEC+1)\n\tld (W),hl\n\tld hl,CHAIN\n\tld (VEC+1),hl\n\tcall VEC\n\tret\n"
    "CHAIN:\n\tld hl,(W)\n\tjp (hl)\n",
])
def test_a_jump_whose_operand_is_read_keeps_it(src):
    """`ld hl,(VEC+1)' reads the operand of `jp HANDLER', and HANDLER is
    `jp REAL': jump threading made VEC `jp REAL', and the program read REAL
    where it had read HANDLER."""
    out = optimize(src + VEC)
    assert "jp HANDLER" in instrs(out), out


@pytest.mark.parametrize("go", [
    "GO:\n\tjp VEC\n",
    "GO:\n\tjr VEC\n",
    # a tail call, `jp VEC'
    "GO:\n\tcall VEC\n\tret\n",
])
@pytest.mark.parametrize("patch", [
    "\tld hl,MINE\n\tld (VEC+1),hl\n",
    "\tld hl,VEC+1\n\tld de,MINE\n\tld (hl),e\n\tinc hl\n\tld (hl),d\n",
])
def test_no_jump_is_threaded_through_a_jump_the_program_patches(patch, go):
    """The program writes MINE over VEC's operand: VEC goes to MINE, and
    so does GO.  Jump threading took VEC to go on to REAL, through
    HANDLER, and made GO's jump `jp REAL'.  (0.2.6 removed VEC's `jp
    HANDLER' instead, as a jump to the next line.)"""
    src = patch + "\tcall GO\n\tld (V),a\n\tjp 0\n" + go + "\tnop\n" + VEC
    out = optimize(src)
    assert [line for line in instrs(out) if line in ("jp VEC", "jr VEC", "call VEC")], out
    assert "jp HANDLER" in instrs(out), out


# HANDLER writes the carry; MINE, which the program writes over the operand of
# the jump or call, reads it.
FLAG_VEC = ("HANDLER:\n\tor 1\n\tld a,2\n\tret\nMINE:\n\tld a,3\n\tret nc\n\tld a,4\n\tret\n")


@pytest.mark.parametrize("src", [
    "\tld hl,MINE\n\tld (VEC+1),hl\n\tcall GO\n\tld (V),a\n\tjp 0\n"
    "GO:\n\tscf\n\tld a,0\n\tjp VEC\n\tnop\nVEC:\n\tjp HANDLER\n",
    "\tld hl,MINE\n\tld (CL+1),hl\n\tcall GO\n\tld (V),a\n\tjp 0\n"
    "GO:\n\tscf\n\tld a,0\nCL:\n\tcall HANDLER\n\tret\n",
])
def test_where_a_jump_or_call_the_program_patches_goes_is_not_known(src):
    """The carry is read where GO goes, at MINE: liveness followed the jump,
    or the call, to HANDLER, where the carry is written first, and `ld
    a,0' became `xor a'."""
    out = optimize(src + FLAG_VEC)
    assert "ld a,0" in instrs(out), out


@pytest.mark.parametrize("patch", [
    "\tld a,0C9h\n\tld (SW),a\n",  # `ret'
    "\txor a\n\tld (SW),a\n",  # `nop'
    "\tld hl,0C9C9h\n\tld (SW),hl\n",  # and `ret' over the `ret' after it
])
def test_liveness_reads_everything_at_an_instruction_the_program_patches(patch):
    """The program writes `ret', or `nop', over SW's `or a': SUB returns
    with the carry its caller set, which the caller reads.  Liveness took
    `or a' to write the carry first, and SUB's `ld a,0' became `xor a'."""
    src = (patch + "\tscf\n\tcall SUB\n\tld a,0\n\tjr nc,N\n\tinc a\nN:\n\tld (V),a\n\tjp 0\n"
           "SUB:\n\tld a,0\nSW:\n\tor a\n\tret\n")
    out = instrs(optimize(src))
    assert out[out.index("or a") - 1] == "ld a,0", out


# ---- Code an instruction the program writes over goes on into -----------------
#
# The program may write over an instruction so that it goes on to the line
# after it, where the text jumps or returns: `ret' made `nop' (0) or `ret nz'
# (0C0h), `jp' made three `nop's, `jp nz' (0C2h) or `ld hl,nn' (21h, which
# skips the operand, as 8080 code does), `jr' made two `nop's or given a
# displacement of 0.  z80sim does not put the bytes of code in memory: these
# run the code as the program makes it (assert_equivalent's `patch').

# QQ pushes SHOWP's argument, and SW goes to SKIP, which takes it off; the
# program writes over SW, so that QQ goes on to `call SHOWP / ret'.
FALL = ("START:\n{patch}\tld bc,5A5Ah\n\tcall QQ\n\tld hl,0\n\tld de,0\n\tjp 0\n"
        "QQ:\n\tpush bc\n{before}SW:\n\t{sw}\n{between}\tcall SHOWP\n\tret\n"
        "SKIP:\n\tpop bc\n\tret\n")


@pytest.mark.parametrize("sw, patch, becomes, before, between", [
    # three `nop's over `jp'
    ("jp SKIP", "\tld hl,0\n\tld (SW+1),hl\n\txor a\n\tld (SW),a\n", "nop\n\tnop\n\tnop", "", ""),
    ("jp SKIP", "\tld hl,0\n\tld (SW),hl\n\txor a\n\tld (SW+2),a\n", "nop\n\tnop\n\tnop", "",
     "\tld a,1\n"),
    # two `nop's over `jr'
    ("jr SKIP", "\tld hl,0\n\tld (SW),hl\n", "nop\n\tnop", "", ""),
    # `jr' to the next line
    ("jr SKIP", "\txor a\n\tld (SW+1),a\n", "jr $+2", "", "\tld a,1\n"),
    # `jp' made `ld hl,nn'
    ("jp SKIP", "\tld a,21h\n\tld (SW),a\n", "ld hl,SKIP", "", ""),
    # `jp' made `jp nz', where Z is set
    ("jp SKIP", "\tld a,0C2h\n\tld (SW),a\n", "jp nz,SKIP", "\txor a\n", ""),
    # `ret' made `nop'
    ("ret", "\txor a\n\tld (SW),a\n", "nop", "", "\tld a,1\n"),
])
def test_no_tail_call_in_code_an_instruction_the_program_writes_over_goes_on_into(
        sw, patch, becomes, before, between):
    """`call SHOWP / ret' counted as reached by nothing, and became `jp
    SHOWP', with QQ's argument still pushed: SHOWP took QQ's return address
    for it.  (Where SW's bytes, or the word the program writes, reach the
    line after SW, that line was already left as it was.)"""
    src = FALL.format(patch=patch, sw=sw, before=before, between=between)
    out = assert_equivalent(src, entry=SHOWP, patch={"SW": becomes})
    assert "call SHOWP" in instrs(out), out


def test_no_tail_call_after_a_jump_of_an_exported_vector():
    """Another module may write over a jump after a label the text exports:
    here three `nop's over HOOK's `jp SKIP', and QQ goes on to `call SHOWP /
    ret' with SHOWP's argument pushed, as above."""
    src = SHOWP_MAIN + "QQ:\n\tpush bc\nHOOK::\n\tjp SKIP\n\tcall SHOWP\n\tret\nSKIP:\n\tpop bc\n\tret\n"
    entry = "\tld hl,0\n\tld (HOOK+1),hl\n\txor a\n\tld (HOOK),a\n" + SHOWP
    out = assert_equivalent(src, entry=entry, patch={"HOOK": "nop\n\tnop\n\tnop"})
    assert "call SHOWP" in instrs(out), out


# START pushes a word and calls QQ, which calls HOOK; the program writes over
# HOOK's `ret', and HOOK goes on to read the word above its return address.
HOOK = ("START:\n{patch}\tld bc,1234h\n\tpush bc\n\tcall QQ\n\tpop bc\n\tld hl,0\n\tld de,0\n"
        "\tjp 0\nQQ:\n{before}\tcall HOOK\n\tret\nHOOK:\n\tret\n\tpop hl\n\tpop de\n\tpush de\n"
        "\tpush hl\n\tld a,d\n\tld (V),a\n\tret\n")


@pytest.mark.parametrize("patch, becomes, before", [
    ("\txor a\n\tld (HOOK),a\n", "nop", ""),
    # `ret nz', where Z is set
    ("\tld a,0C0h\n\tld (HOOK),a\n", "ret nz", "\txor a\n"),
])
def test_no_tail_call_to_a_routine_whose_ret_the_program_writes_over(patch, becomes, before):
    """HOOK counted as a routine that returns to its call, and `call HOOK /
    ret' became `jp HOOK': jumped to, HOOK read START's return address
    where it had read the word START pushed."""
    src = HOOK.format(patch=patch, before=before)
    out = assert_equivalent(src, patch={"HOOK": becomes})
    assert "call HOOK" in instrs(out), out
