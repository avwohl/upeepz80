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
