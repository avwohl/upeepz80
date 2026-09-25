"""Storage the text defines: where it is, and what can read it."""

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
