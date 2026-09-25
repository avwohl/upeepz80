"""Storage the text defines: where it is, and what can read it.

Dead-store elimination drops the store of a parameter at a procedure's
entry when nothing can read the byte afterwards.  Something can read it
through an address computed from a neighbour's, as PL/M-80's `.a + 1'
does, and not only by its name.
"""

import pytest

from upeepz80 import optimize

from tests._equiv import assert_equivalent, instrs
from tests.z80sim import DATA_BASE, Machine


def test_the_interpreter_lays_out_data_as_an_assembler_does():
    """Each line of data follows the one before, as long as the assembler
    makes it, whatever segment or code lies between; `db' and `dw' fill
    it before the run; a label alone on its line is the next line's."""
    src = ("\tld hl,B+1\n\tld a,(hl)\n\tld de,(W)\n\tret\n"
           "\tdseg\nA:\tds 2\nB:\n\tdb 'xy',7\n\tcseg\nC:\n\tnop\n\tdseg\nW:\tdw B\n"
           "E\tequ W+1\n")
    m = Machine(src)
    assert m.symbols["a"] == DATA_BASE
    assert m.symbols["b"] == DATA_BASE + 2
    assert m.symbols["w"] == DATA_BASE + 5
    assert m.symbols["e"] == DATA_BASE + 6
    assert m.labels["c"] == src.split("\n").index("C:")
    assert m.run() == "ret"
    assert m.r["a"] == ord("y")
    assert (m.r["d"] << 8 | m.r["e"]) == DATA_BASE + 2


# ---- the program uplm80's storage work found -----------------------------------

# uplm80 0.3.7 (without its workaround) at -O2, what it hands the optimizer
# for
#
#     pq: procedure (a, b) byte;
#         declare (a, b) byte;
#         declare pp address, c based pp byte;
#         pp = .a + 1;
#         return c;
#     end pq;
#     call mon1(2, pq('A', 'B'));
#
# less the three lines that need CP/M: `ld hl,(6) / ld sp,hl' and `call 5'.
# pq's result is left in E.
PQ = """\
\t.z80
\tcseg

\torg\t0100H


; Module initialization code
\tld\ta,41H
\tld\t(@PQ$@A),a
\tld\ta,42H
\tcall\tPQ
\tld\te,a
\tld\tc,2
\tjp\t0
\textrn\tMON1

; Procedure PQ
PQ:
\tld\t(@PQ$@B),a
\tld\thl,@PQ$@A
\tinc\thl
\tld\t(@PQ$PP),hl
\tld\thl,(@PQ$PP)
\tld\ta,(hl)
\tret

\tdseg
; Data segment

; Variables
@PQ$@A:\tds\t1
@PQ$@B:\tds\t1
@PQ$PP:\tds\t2

\tend
"""


def test_a_parameter_reached_through_the_one_before_it():
    """`b' arrives in A and is stored at pq's entry, and nothing names its
    storage after that; `pp = .a + 1' reaches it from `a''s.  The store
    was dropped, and pq('A', 'B') returned whatever the byte held."""
    out = assert_equivalent(PQ)
    assert "ld (@PQ$@B),a" in instrs(out)
    for src in (PQ, out):
        m = Machine(src)
        assert m.run() == "boot"
        assert m.r["e"] == ord("B")


# ---- the ways to a byte from a neighbour's address ------------------------------

def program(body: str, data: str = "") -> str:
    """P stores A in PB at its entry; ``body`` follows the store."""
    return ("\tld a,41h\n\tld (PA),a\n\tld a,42h\n\tcall P\n\tjp 0\nP:\n\tld (PB),a\n" + body +
            "\tret\n\tdseg\nPA:\tds 1\nPB:\tds 1\nPW:\tds 2\n" + data)


@pytest.mark.parametrize("body,data", [
    # A pointer made from the byte before, or after.
    ("\tld hl,PA\n\tinc hl\n\tld a,(hl)\n", ""),
    ("\tld hl,PA+1\n\tld a,(hl)\n", ""),
    ("\tld de,PW\n\tdec de\n\tld a,(de)\n", ""),
    ("\tld hl,PA\n\tld de,1\n\tadd hl,de\n\tld e,(hl)\n", ""),
    # A table holds the neighbour's address.
    ("\tld hl,(TAB)\n\tinc hl\n\tld a,(hl)\n", "TAB:\tdw PA\n"),
    # Another name for the neighbour, or for the byte itself.
    ("\tld hl,NEXT\n\tld a,(hl)\n", "NEXT\tequ PA+1\n"),
    ("\tld a,(NEXT)\n", "NEXT\tequ PA+1\n"),
    ("\tld hl,ALIAS\n\tinc hl\n\tld a,(hl)\n", "ALIAS\tequ PA\n"),
    # A load at an address spelled from the neighbour's name.
    ("\tld a,(PA+1)\n", ""),
    ("\tld a,(PW-1)\n", ""),
    ("\tld hl,(PA)\n\tld a,h\n", ""),
    ("\tld a,(PA + 1)\n", ""),
    ("\tld a,(PA+2-1)\n", ""),
])
def test_a_store_a_neighbours_address_reaches_is_kept(body, data):
    out = assert_equivalent(program(body, data))
    assert "ld (PB),a" in instrs(out)


@pytest.mark.parametrize("src", [
    # The neighbour's address, handed to code elsewhere.
    program("\tld hl,PA\n\tcall ELSE\n"),
    program("\tld hl,PA\n\tpush hl\n"),
    program("\tld hl,PA\n\tld (PTR),hl\n", "PTR:\tds 2\n"),
    program("\tld (PTR),hl\n\tld hl,PA\n\tret\n", "PTR:\tds 2\n"),
    # Exported: another module can compute the address from it.
    "\tpublic PA\n" + program(""),
    program("").replace("PA:", "PA::"),
    program("", "\tpublic PW\n"),
    # Its address in a register other than HL, or in a byte.
    program("\tld bc,PA\n"),
    program("\tld a,LOW(PA)\n"),
    program("", "\tdb HIGH(PA)\n"),
    # `$' where the data is.
    program("\tld hl,HERE\n", "HERE\tequ $\n"),
    program("", "\tdw $\n"),
    # SP pointed there: pops read it.
    program("\tld sp,PW\n"),
    # Set more than once, to the neighbour among others.
    program("\tld hl,VAR\n", "VAR\tdefl PW\nVAR\tdefl 0\n"),
    # A segment the text places: a number may be its address.
    program("", "\torg 200h\n"),
    # A call's return address is the address of what follows it: here the
    # byte itself, the call's inline argument.
    ("\tld a,42h\n\tcall SETP\n\tcall SHOW\nPARM:\tdb 0\n\tjp 0\n"
     "SETP:\n\tld (PARM),a\n\tret\nSHOW:\n\tpop hl\n\tld a,(hl)\n\tinc hl\n\tpush hl\n\tret\n"),
])
def test_a_store_an_address_that_leaves_the_text_may_reach_is_kept(src):
    out = optimize(src)
    assert "ld (PB),a" in instrs(out) or "ld (PARM),a" in instrs(out), out


@pytest.mark.parametrize("body,data", [
    # Nothing reads it, or what is read is another byte.
    ("", ""),
    ("\tld a,(PA)\n\tld hl,(PW)\n", ""),
    ("\tld (PA),a\n\tld (PW),hl\n", ""),
    # Numbers and other modules' names are not addresses in the segment,
    # nor is the address of code.
    ("\tld hl,8000h\n\tld a,(8001h)\n", ""),
    ("\tld hl,OTHER+1\n", "\textrn OTHER\n"),
    ("\tld hl,P\n", ""),
    # Storage across code from the byte: an address computed across code
    # would change with every rewrite that shortens it.
    ("\tld hl,CA\n\tld a,(CA+0)\n", "\tcseg\nCA:\tds 1\n"),
])
def test_a_store_nothing_can_reach_still_goes(body, data):
    src = program(body, data)
    assert "ld (PB),a" not in instrs(optimize(src)), src


def test_a_call_followed_by_code_gives_no_address_in_the_data():
    """The return address of `call P1' is the `ret' after it, code: no
    storage is reached from it."""
    src = "\tcall P1\n\tret\nP1:\n\tld (SLOT),a\n\tret\nSLOT:\tds 1\n"
    assert "(SLOT),a" not in optimize(src)
    src = "\tcall P1\nSLOT:\tds 1\nP1:\n\tld (SLOT),a\n\tret\n"
    assert "(SLOT),a" in optimize(src)


def test_nothing_is_dead_where_the_text_is_not_all_there():
    """A conditional, a macro, an include, a name defined twice or an
    instruction the optimizer does not know: the layout is not known."""
    for extra in ("\tif 1\n\tendif\n", "\tinclude x.lib\n", "M\tmacro\n\tendm\n",
                  "PB:\tds 1\n", "\tfoo 1\n"):
        src = program("") + extra
        assert "ld (PB),a" in instrs(optimize(src)), extra
