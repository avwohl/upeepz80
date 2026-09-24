"""Rewrites that change a register or flag happen only where it is dead.

Most cases run the code before and after optimization on tests/z80sim.py
(see tests/_equiv.py); the rest check where a rewrite is and is not made.
"""

import pytest

from upeepz80 import optimize

from tests._equiv import assert_equivalent, instrs


# ---- two of the defects uplm80's differential test found --------------------------

def test_inc_in_memory_keeps_a_when_a_is_read():
    """uplm80's `b, w = -(NOT b)': after cpl/cpl and the push/pop around the
    store go, `ld a,(x) / inc a / ld (x),a' was made `ld hl,x / inc (hl)',
    and w got A's old value (and HL was clobbered too)."""
    src = ("\tld\ta,(@B)\n\tcpl\n\tcpl\n\tinc\ta\n\tpush\taf\n\tld\t(@B),a\n\tpop\taf\n"
           "\tld\tl,a\n\tld\th,0\n\tld\t(W),hl\n\tjp\t0\n")
    out = assert_equivalent(src)
    assert "inc (hl)" not in instrs(out)


def test_inc_in_memory_where_a_and_hl_are_dead():
    src = "\tld a,(COUNT)\n\tinc a\n\tld (COUNT),a\n\tld a,1\n\tld hl,0\n\tret\n"
    out = assert_equivalent(src)
    assert instrs(out)[:2] == ["ld hl,COUNT", "inc (hl)"]


def test_ld_hl_const_kept_when_a_store_of_hl_reads_it():
    """`W = (B := 9)': the store of HL is a read of HL, and `ld hl,9 / ld a,l'
    was made `ld a,9', storing whatever HL held."""
    src = ("\tld\thl,9\n\tld\ta,l\n\tld\t(@B),a\n\tld\t(W),hl\n\tld\thl,3\n\tld\t(V),hl\n"
           "\tjp\t0\n")
    out = assert_equivalent(src)
    assert "ld a,9" not in instrs(out)


def test_ld_hl_const_kept_after_push_store_pop():
    """The shape the uplm80 CHANGELOG records for `w1, w0 = (sb := ...)'."""
    src = ("\tld\thl,0\n\tld\ta,l\n\tld\t(SB),a\n\tpush\thl\n\tld\t(W1),hl\n\tpop\thl\n"
           "\tld\t(W0),hl\n\tld\thl,5\n\tjp\t0\n")
    assert_equivalent(src)


def test_low_of_a_constant_stored_as_address():
    """LOW(SIZE(aw)) stored to an ADDRESS (difftest seed 80071)."""
    src = ("\tld\thl,16\n\tld\ta,l\n\tld\tl,a\n\tld\th,0\n\tld\t(W4),hl\n\tld\thl,5\n"
           "\tjp\t0\n")
    assert_equivalent(src)


# ---- the rest of the audit -------------------------------------------------------

@pytest.mark.parametrize("src", [
    # ld a,0 -> xor a clears the carry rla, adc and jp c read
    "\tscf\n\tld a,0\n\trla\n\tret\n",
    "\tscf\n\tld a,0\n\tadc a,b\n\tld (V),a\n\tld a,1\n\tcp b\n\tret\n",
    "\tscf\n\tld a,0\n\tjp c,L1\n\tld b,1\nL1:\n\tret\n",
    "\tscf\n\tld a,0\n\tpush af\n\tpop hl\n\tld (V),hl\n\tcp b\n\tret\n",
    # cp 0 -> or a differs in P/V and N
    "\tcp 0\n\tjp pe,L1\n\tld b,1\nL1:\n\tcp b\n\tret\n",
    # inc a / dec a leaves the dec's flags
    "\tinc a\n\tdec a\n\tjr z,L1\n\tld b,1\nL1:\n\tcp b\n\tret\n",
    # cpl / cpl and ccf / ccf set H and N, which daa reads
    "\tld a,15h\n\tadd a,27h\n\tcpl\n\tcpl\n\tdaa\n\tld (V),a\n\tcp b\n\tret\n",
    "\tld a,15h\n\tsub 27h\n\tccf\n\tccf\n\tdaa\n\tld (V),a\n\tcp b\n\tret\n",
    # and 0ffh sets H, or a clears it
    "\tld a,19h\n\tand 0ffh\n\tdaa\n\tld (V),a\n\tcp b\n\tret\n",
    # ld a,(hl) / ld e,a -> ld e,(hl) loses A
    "\tld hl,V\n\tld a,(hl)\n\tld e,a\n\tld (W),a\n\tret\n",
    # ld h,0 / ld d,h / ld e,l -> ld d,0 / ld e,l loses H = 0
    "\tld h,0\n\tld d,h\n\tld e,l\n\tld (W),hl\n\tret\n",
    # ld l,a / ld h,0 / sub l: the sub reads L
    "\tld a,7\n\tld l,a\n\tld h,0\n\tsub l\n\tld (V),a\n\tld hl,0\n\tcp b\n\tret\n",
    "\tld hl,V\n\tld l,a\n\tld h,0\n\tcp (hl)\n\tld hl,0\n\tret\n",
    # ld a,l / ld h,0 / ld (hl),a: the store's address reads H
    "\tld hl,V\n\tld a,l\n\tld h,0\n\tld (hl),a\n\tld hl,0\n\tret\n",
    "\tld hl,V\n\tld l,a\n\tld h,0\n\tld (hl),a\n\tld hl,0\n\tret\n",
    # ld hl,0 / ld a,l / ld (hl),a stores to 0, not to old HL
    "\tld hl,0\n\tld a,l\n\tld (hl),a\n\tld hl,5\n\tcp b\n\tret\n",
    # ... and xor a sets the flags the original leaves
    "\tscf\n\tld hl,0\n\tld a,l\n\tld (V),a\n\tjp c,L1\n\tld b,1\nL1:\n\tret\n",
    # ld de,1 / add hl,de -> inc hl: the carry, and DE
    "\tld hl,0ffffh\n\tld de,1\n\tadd hl,de\n\tjp c,L1\n\tld b,1\nL1:\n\tld de,0\n\tret\n",
    "\tld de,1\n\tadd hl,de\n\tld (W),de\n\tcp b\n\tret\n",
    # ld de,2 / call ??mul16 -> add hl,hl: the routine leaves DE = 0 and A = 0
    "\tld de,2\n\tcall ??mul16\n\tld (W),de\n\tret\n" + "??mul16:\n\tld b,h\n\tld c,l\n\tld hl,0\n"
    "??L:\n\tld a,e\n\tor d\n\tret z\n\tsrl d\n\trr e\n\tjp nc,??S\n\tadd hl,bc\n"
    "??S:\n\tsla c\n\trl b\n\tjp ??L\n",
    # ld de,0 / call ??subde: the routine sets Z, and clears C
    "\tscf\n\tld de,0\n\tcall ??subde\n\tjp c,L1\n\tld b,1\nL1:\n\tret\n??subde:\n\tor a\n\tsbc hl,de\n\tret\n",
    # the shift leaves A = new L
    "\tor a\n\tld a,h\n\trra\n\tld h,a\n\tld a,l\n\trra\n\tld l,a\n\tld (V),a\n\tcp b\n\tret\n",
    # dec b / jp nz -> djnz: the flags dec b sets are read where the loop exits
    "\tld b,3\nL1:\n\tinc c\n\tdec b\n\tjp nz,L1\n\tjp z,L2\n\tld c,0\nL2:\n\tret\n",
    # ... and where it goes back
    "\tld b,3\n\tld c,0\nL1:\n\tjr z,L3\n\tld c,5\nL3:\n\tdec b\n\tjp nz,L1\n\tcp b\n\tret\n",
    # a ret at a stack depth other than the routine's goes where the stack says
    "\tscf\n\tld a,0\n\tcall P\n\tcp b\n\tjp 0\nP:\n\tld hl,Q\n\tpush hl\n\tret\nQ:\n\trla\n\tret\n",
    # a conditional call that is not made leaves HL as it was
    "\tld hl,5\n\tld a,l\n\tcall z,P\n\tld (W),hl\n\tjp 0\nP:\n\tld hl,0\n\tret\n",
    # the flags a routine leaves are read after the call
    "\tcall P\n\tjr c,X\n\tld b,1\nX:\n\tjp 0\nP:\n\tscf\n\tld a,0\n\tret\n",
    # ... by one caller of two
    "\tcall P\n\tcp b\n\tcall P\n\tjr c,X\n\tld b,1\nX:\n\tjp 0\nP:\n\tscf\n\tld a,0\n\tret\n",
    # a routine reads the flags on entry
    "\tscf\n\tld a,0\n\tcall P\n\tcp b\n\tjp 0\nP:\n\trla\n\tld (V),a\n\tret\n",
    # the flags pushed with A come back with it
    "\tscf\n\tld a,0\n\tpush af\n\tld a,5\n\tpop af\n\trla\n\tjp 0\n",
    # a rewrite starting at a label with an instruction after it kept the
    # instruction and lost the label
    "\tjp L1\nL0:\tld a,0\n\tret\nL1:\tld a,5\n\tjp L0\n",
])
def test_every_rewrite_keeps_what_is_read(src):
    assert_equivalent(src)


def test_no_pattern_across_data():
    """Data between two instructions is not a comment: `push hl / db 5 /
    pop hl' runs the byte in between."""
    out = optimize("\tpush hl\n\tdb 5\n\tpop hl\n\tret\n")
    assert instrs(out) == ["push hl", "db 5", "pop hl", "ret"]


def test_ld_a_0_is_xor_a_where_the_flags_are_dead():
    assert instrs(optimize("\tld a,0\n\tcp b\n\tret\n"))[0] == "xor a"
    assert instrs(optimize("\tld a,0\n\tret\n"))[0] == "ld a,0"


def test_liveness_follows_branches_and_loops():
    # The flags are read at the branch target, not on the fall-through.
    src = "\tld a,0\n\tjp L1\n\tcp b\nL1:\n\tjr c,L2\nL2:\n\tcp b\n\tret\n"
    assert instrs(optimize(src))[0] == "ld a,0"
    # Round a loop and out: every path writes the flags before it reads them.
    src = "\tld a,0\nL1:\n\tcp b\n\tjr nz,L1\n\tret\n"
    assert instrs(optimize(src))[0] == "xor a"


def test_liveness_stops_at_what_it_cannot_follow():
    # What happens out of the module, or at an address not known here, may
    # read anything.
    for tail in ("\tcall ELSEWHERE\n\tcp b\n\tret\n", "\tjp (hl)\n", "\tjp ELSEWHERE\n",
                 "\tdb 0\n", "", "\trst 38h\n", "\tret\n"):
        assert instrs(optimize("\tld a,0\n" + tail))[0] == "ld a,0", tail


def test_liveness_follows_calls_and_returns():
    # Into the routine and back: the flags are written after the call.
    src = "\tld a,0\n\tcall S\n\tcp b\n\tjp 0\nS:\n\tld (V),a\n\tret\n"
    assert instrs(optimize(src))[0] == "xor a"
    # The routine reads the carry.
    src = "\tld a,0\n\tcall S\n\tcp b\n\tjp 0\nS:\n\trla\n\tret\n"
    assert instrs(optimize(src))[0] == "ld a,0"
    # Back from a ret to every caller: one of them reads the flags.
    src = ("\tcall P\n\tcp b\n\tcall P\n\tjr z,X\nX:\n\tjp 0\n"
           "P:\n\tld a,0\n\tret\n")
    assert "ld a,0" in instrs(optimize(src))
    src = ("\tcall P\n\tcp b\n\tcall P\n\tcp c\n\tjp 0\n"
           "P:\n\tld a,0\n\tret\n")
    assert "xor a" in instrs(optimize(src))
    # A routine whose address is taken may be entered from anywhere.
    src = ("\tld hl,P\n\tcall P\n\tcp b\n\tjp 0\n"
           "P:\n\tld a,0\n\tret\n")
    assert "ld a,0" in instrs(optimize(src))
    # So may one the module exports.
    src = ("\tpublic P\n\tcall P\n\tcp b\n\tjp 0\n"
           "P:\n\tld a,0\n\tret\n")
    assert "ld a,0" in instrs(optimize(src))


def test_liveness_follows_a_value_through_the_stack():
    # push af / pop af puts back the flags: the ones pushed are read if the
    # ones popped are.
    src = "\tld a,0\n\tpush af\n\tld b,a\n\tpop af\n\tjr c,X\nX:\n\tcp b\n\tjp 0\n"
    assert instrs(optimize(src))[0] == "ld a,0"
    src = "\tld a,0\n\tpush af\n\tld b,a\n\tpop af\n\tcp b\n\tjp 0\n"
    assert instrs(optimize(src))[0] == "xor a"
    # Pushed as AF and popped as HL: the flags are in L.
    src = "\tld a,0\n\tpush af\n\tpop hl\n\tld (W),hl\n\tcp b\n\tjp 0\n"
    assert instrs(optimize(src))[0] == "ld a,0"
    # A routine that takes its return address off the stack.
    src = ("\tld a,0\n\tcall P\n\tcp b\n\tjp 0\n"
           "P:\n\tpop hl\n\tpush hl\n\tret\n")
    assert_equivalent(src)
    # ex de,hl moves the value it is asked about.
    src = "\tld hl,5\n\tld a,l\n\tex de,hl\n\tld (W),de\n\tret\n"
    assert instrs(optimize(src))[0] == "ld hl,5"
    src = "\tld hl,5\n\tld a,l\n\tex de,hl\n\tld de,0\n\tld hl,0\n\tret\n"
    assert instrs(optimize(src))[0] == "ld a,5"


def test_dead_store_named_by_an_equ_is_kept():
    """`ALIAS: EQU PARAM' (uplm80's form for AT) names the location, and a
    read of ALIAS reads it."""
    for equ in ("ALIAS:\tEQU\tPARAM", "ALIAS\tequ\tPARAM"):
        src = equ + "\nP:\n\tld (PARAM),a\n\tret\nQ:\n\tld a,(ALIAS)\n\tret\n"
        assert "(PARAM),a" in optimize(src), equ
