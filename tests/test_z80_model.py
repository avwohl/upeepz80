"""The instruction model: what each instruction reads and writes, and its size."""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from upeepz80.z80 import ALL, FLAGS, UNKNOWN, data_size, effect, strip_comment, split_operands

R8 = ["a", "b", "c", "d", "e", "h", "l"]
ALU = ["add a,", "adc a,", "sub ", "sbc a,", "and ", "xor ", "or ", "cp "]
CC = ["nz", "z", "nc", "c", "po", "pe", "p", "m"]


def forms() -> list[str]:
    """Every instruction form the model knows, once."""
    out = []
    out += [f"ld {d},{s}" for d in R8 for s in R8]
    out += [f"ld {r},12h" for r in R8] + [f"ld {r},(hl)" for r in R8] + [f"ld (hl),{r}" for r in R8]
    out += [f"ld {r},(ix+5)" for r in R8] + [f"ld (iy-3),{r}" for r in R8]
    out += ["ld (hl),7", "ld (ix+1),7", "ld a,(bc)", "ld a,(de)", "ld (bc),a", "ld (de),a",
            "ld a,(V)", "ld (V),a", "ld a,i", "ld a,r", "ld i,a", "ld r,a"]
    for rr in ("bc", "de", "hl", "sp", "ix", "iy"):
        out += [f"ld {rr},1234h", f"ld {rr},(V)", f"ld (V),{rr}"]
    out += ["ld sp,hl", "ld sp,ix", "ld sp,iy"]
    for rr in ("bc", "de", "hl", "af", "ix", "iy"):
        out += [f"push {rr}", f"pop {rr}"]
    out += ["ex de,hl", "ex af,af'", "exx", "ex (sp),hl", "ex (sp),ix", "ex (sp),iy"]
    for op in ALU:
        out += [f"{op}{s}" for s in R8 + ["(hl)", "(ix+2)", "0f0h"]]
    for rr in ("bc", "de", "hl", "sp"):
        out += [f"add hl,{rr}", f"adc hl,{rr}", f"sbc hl,{rr}"]
    out += ["add ix,bc", "add ix,de", "add ix,ix", "add ix,sp", "add iy,iy"]
    for op in ("inc", "dec"):
        out += [f"{op} {r}" for r in R8 + ["(hl)", "(ix+3)", "bc", "de", "hl", "sp", "ix", "iy"]]
    out += ["daa", "cpl", "neg", "ccf", "scf", "nop", "halt", "di", "ei", "im 1",
            "rlca", "rrca", "rla", "rra", "rld", "rrd"]
    for op in ("rlc", "rrc", "rl", "rr", "sla", "sra", "srl"):
        out += [f"{op} {r}" for r in R8 + ["(hl)", "(ix+4)"]]
    for op in ("bit", "set", "res"):
        out += [f"{op} 3,{r}" for r in R8 + ["(hl)", "(iy+4)"]]
    out += ["jp L0", "jp (hl)", "jp (ix)", "jp (iy)", "jr L0", "djnz L0", "call L0", "ret",
            "reti", "retn", "rst 38h"]
    out += [f"jp {c},L0" for c in CC] + [f"call {c},L0" for c in CC] + [f"ret {c}" for c in CC]
    out += [f"jr {c},L0" for c in CC[:4]]
    out += ["in a,(10h)", "in b,(c)", "out (10h),a", "out (c),e"]
    out += ["ldi", "ldir", "ldd", "lddr", "cpi", "cpir", "cpd", "cpdr",
            "ini", "inir", "ind", "indr", "outi", "otir", "outd", "otdr"]
    return out


def model(text: str):
    op, _, operands = text.partition(" ")
    return effect(op, operands)


def test_every_form_is_known():
    unknown = [f for f in forms() if model(f) is UNKNOWN]
    assert not unknown


@pytest.mark.parametrize("text", [
    "ld hl,ix+5", "ld hl,hl", "ld a,299", "cp 256", "ld (hl),(hl)", "ld a,(ix+200)",
    "jr pe,L0", "ld (ix+1),hl", "add hl,ix", "ld ixh,5", "ex hl,de", "push sp", "inc af",
])
def test_not_z80_instructions(text):
    assert model(text) is UNKNOWN


def test_reads_and_writes():
    e = model("ld (W),hl")
    assert e.reads == {"h", "l"} and not e.writes
    e = model("add hl,de")
    assert e.writes == {"h", "l", "fh", "fn", "fc"}
    e = model("inc a")
    assert "fc" not in e.writes and "fz" in e.writes
    assert model("xor a").reads == frozenset()          # A ^ A does not depend on A
    assert model("sbc a,a").reads == {"fc"}
    assert model("bit 0,a").writes == {"fz", "fh", "fn"}   # S and P/V are undefined
    # A call or return reads its condition; what the callee or caller reads
    # is found by following it.
    assert model("call X").reads == frozenset() and model("ret z").reads == {"fz"}
    assert model("call X").flow == "call" and model("ret z").cond
    assert model("jp (hl)").reads == ALL
    assert model("djnz LOOP").reads == {"b"} and model("djnz LOOP").flow == "branch"
    assert model("djnz L").reads == ALL   # a label named like a register is not followed
    assert model("ex de,hl").swap_de_hl
    assert model("push af").reads >= FLAGS | {"a"}
    assert model("ld a,(ix-3)").reads == {"ixh", "ixl"}


def test_comments_and_operands():
    assert strip_comment("\tld a,';' ; x") == "\tld a,';'"
    assert strip_comment("\tex af,af' ; swap") == "\tex af,af'"
    assert split_operands("a,(ix+1)") == ["a", "(ix+1)"]
    assert split_operands("'a,b',2") == ["'a,b'", "2"]


def _um80():
    return shutil.which("um80")


@pytest.mark.skipif(_um80() is None, reason="um80 is not installed")
def test_sizes_agree_with_the_assembler():
    """Relative jumps are only made where the model's sizes say they reach,
    so the sizes have to be the assembler's."""
    # Relative jumps first, near their target: um80 makes one that does not
    # reach a jp (and says so) rather than fail.
    body = sorted(forms(), key=lambda f: not f.startswith(("jr", "djnz")))
    src = "\t.z80\nV\tequ\t8000h\nL0:\n" + "".join(f"\t{f}\n" for f in body) + "\tend\n"
    with tempfile.TemporaryDirectory() as d:
        mac, prn, rel = (os.path.join(d, n) for n in ("S.MAC", "S.PRN", "S.REL"))
        with open(mac, "w") as fh:
            fh.write(src)
        r = subprocess.run(["um80", mac, "-l", prn, "-o", rel], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        listing = open(prn).read().split("\n")
    sizes = {}
    for row in listing:
        m = re.match(r"^\s*(\d+)\s+([0-9A-F]{4})\s+((?:[0-9A-F]{2}'?\s)+)", row)
        if m:
            sizes[int(m.group(1))] = len(m.group(3).split())
    wrong = []
    for n, f in enumerate(body, start=4):
        if sizes.get(n) != model(f).size:
            wrong.append((f, sizes.get(n), model(f).size))
    assert not wrong


# Data lines: the sizes the model gives, and those it does not know.  An
# empty item is one um80 drops at the end of a line and emits a byte or a
# word for elsewhere; another assembler need not do either.
DATA_KNOWN = ["db 1", "db 1,2", "db 'ab','c'", "db 'it''s'", "db \"it's\"", "db 'a,b'",
              "db ''", "db ''''", "db 'a'+'b'", "db 'a' + 1", "db ('ab')", "db +'ab'",
              "defm 'ab','cd'", "defb 1,2", "db 7,'x',8", "dw 1", "dw 1,2", "dw 'a'",
              "defw 1,2", "ds 3", "ds 2,", "db 0 ; ,"]
DATA_UNKNOWN = ["db 1,", "db 1, ", "db 1 ,", "db 1,;c", "db 1,2,", "db 'ab',", "db 'a',,",
                "defb 1,", "defm 'a',", "db 1,,2", "db ,", "dw 1,", "dw ,1", "defw 1,",
                "db 'a' 'b'", "db 'a''"]


def data_model(text: str) -> int | None:
    op, _, operands = strip_comment(text).partition(" ")
    return data_size(op, split_operands(operands))


def test_the_size_of_a_data_line():
    assert [data_model(t) for t in DATA_KNOWN] == [1, 2, 3, 4, 4, 3, 0, 1, 1, 1, 1, 1, 4, 2, 3,
                                                   2, 4, 2, 4, 3, 2, 1]
    assert [t for t in DATA_UNKNOWN if data_model(t) is not None] == []


@pytest.mark.skipif(_um80() is None, reason="um80 is not installed")
def test_data_sizes_agree_with_the_assembler():
    """Dead-store elimination takes a byte's offset from the sizes of the
    data before it, so a size, where the model gives one, is the
    assembler's: every data line is followed by a label, which the linker's
    symbol file gives the address of."""
    lines = [t for t in DATA_KNOWN + DATA_UNKNOWN if t not in ("db 'a' 'b'", "db 'a''")]
    src = "\t.z80\n\tdseg\n" + "".join(f"L{n}::\t{t}\n" for n, t in enumerate(lines))
    src += f"L{len(lines)}::\n\tend\n"
    with tempfile.TemporaryDirectory() as d:
        mac, rel = os.path.join(d, "D.MAC"), os.path.join(d, "D.REL")
        with open(mac, "w") as fh:
            fh.write(src)
        r = subprocess.run(["um80", mac, "-o", rel], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        r = subprocess.run(["ul80", "-s", "-o", os.path.join(d, "D.COM"), rel],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        sym = open(os.path.join(d, "D.SYM")).read()
    at = {name.lower(): int(value, 16)
          for value, name in re.findall(r"([0-9A-F]{4})\s+(L\d+)", sym)}
    wrong = []
    for n, t in enumerate(lines):
        size = data_model(t)
        if size is not None and at[f"l{n + 1}"] - at[f"l{n}"] != size:
            wrong.append((t, at[f"l{n + 1}"] - at[f"l{n}"], size))
    assert not wrong
