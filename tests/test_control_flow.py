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
    src = "P:\n\tld (??AUTO+3),a\n\tret\nQ:\n\tld hl,(??AUTO+2)\n\tret\n"
    assert "(??AUTO+3),a" in optimize(src)
    src = "P:\n\tld (??AUTO+1),a\n\tret\nQ:\n\tld de,(??AUTO)\n\tret\n"
    assert "(??AUTO+1),a" in optimize(src)
    src = "P:\n\tld (??AUTO+3),a\n\tret\nQ:\n\tld a,(??AUTO+2)\n\tret\n"
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