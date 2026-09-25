"""A small Z80 interpreter that runs assembly text, for testing the optimizer.

It executes the source lines themselves - there is no assembler in between -
so that a program and its optimized version can be run from the same state
and their final states compared.  Flags are computed as the Z80 computes
them (Zilog's documented behaviour, plus the parity and overflow cases that
are well established); the undocumented bits 3 and 5 of F are not modelled
and should be masked off when comparing.

Symbols name addresses: pass ``symbols={"V0": 0x8000, ...}``.  ``equ`` lines
in the source are honoured.  The data the source defines (``ds``, ``db``,
``dw``) is laid out from DATA_BASE on as an assembler lays it out, one line
after another; a label of code is only its line.  A ``call`` to a label in
the source pushes a return address and goes there; a ``ret`` with nothing
of ours on the stack ends the run, as do ``halt``, ``jp 0`` and running off
the end.
"""

from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from upeepz80.z80 import data_size, parse_number, split_operands, strip_comment  # noqa: E402

CODE_BASE = 0x1000  # pseudo address of line 0; return addresses are CODE_BASE + line
DATA_BASE = 0xA000  # where the data the text defines (ds, db, dw) is laid out
SENTINEL = 0x0FFE  # the return address the run starts with

DATA_OPS = ("db", "dw", "ds", "defb", "defw", "defs")
# Directives that emit nothing and are passed over when run.
DIRECTIVES = ("public", "extrn", "extern", ".z80", "end", "org", "title", "name",
              "cseg", "dseg", "aseg")

FS, FZ, FH, FP, FN, FC = 0x80, 0x40, 0x10, 0x04, 0x02, 0x01
FLAG_MASK = 0xD7  # S Z H P/V N C

R8 = ("a", "b", "c", "d", "e", "h", "l")
COND = {
    "z": lambda f: bool(f & FZ), "nz": lambda f: not f & FZ,
    "c": lambda f: bool(f & FC), "nc": lambda f: not f & FC,
    "pe": lambda f: bool(f & FP), "po": lambda f: not f & FP,
    "m": lambda f: bool(f & FS), "p": lambda f: not f & FS,
}


class SimError(Exception):
    """The program did something this interpreter does not model."""


def parity(v: int) -> bool:
    return bin(v & 0xFF).count("1") % 2 == 0


def szp(v: int) -> int:
    v &= 0xFF
    f = v & FS
    if v == 0:
        f |= FZ
    if parity(v):
        f |= FP
    return f


class Machine:
    def __init__(self, source: str, symbols: dict[str, int] | None = None):
        self.lines: list[tuple[str, list[str]] | None] = []
        self.labels: dict[str, int] = {}
        self.symbols = {k.lower(): v for k, v in (symbols or {}).items()}
        equates: list[tuple[str, str]] = []
        for idx, raw in enumerate(source.split("\n")):
            text = strip_comment(raw)
            if not text.strip():
                self.lines.append(None)
                continue
            label = None
            body = text
            if not text[0].isspace():
                m = re.match(r"^([A-Za-z_?@$.][\w?@$.]*)(::?)?\s*(.*)$", text)
                if not m:
                    raise SimError(f"cannot parse {raw!r}")
                label, colon, body = m.group(1), m.group(2), m.group(3)
                if not colon:
                    # NAME equ VALUE
                    parts = body.split(None, 1)
                    if parts and parts[0].lower() in ("equ", "defl", "set"):
                        equates.append((label.lower(), parts[1]))
                        self.lines.append(None)
                        continue
                    raise SimError(f"cannot parse {raw!r}")
            body = body.strip()
            if body:
                parts = body.split(None, 1)
                op = parts[0].lower()
                ops = split_operands(parts[1]) if len(parts) > 1 else []
                if op in ("equ", "defl") and label:
                    equates.append((label.lower(), ops[0]))
                    self.lines.append(None)
                    continue
                self.lines.append((op, ops))
            else:
                self.lines.append(None)
            if label:
                if label.lower() in self.labels:
                    raise SimError(f"label {label} defined twice")
                self.labels[label.lower()] = idx
        self._lay_out_data()
        for name, expr in equates:
            self.symbols[name] = self.eval(expr)
        self.mem = bytearray(0x10000)
        self.r = {k: 0 for k in R8}
        self.f = 0
        self.ix = 0
        self.iy = 0
        self.sp = 0xF000
        self.pc = 0
        self.steps = 0

    def _lay_out_data(self) -> None:
        """Give the labels of data their addresses, as an assembler would.

        Code has no bytes here: a label of code is its line.  Data is laid
        out from DATA_BASE on, one line after another in the order of the
        text and as long as the assembler makes it, so that an address
        computed from one label (``ld hl,A+1``) finds what the assembler
        puts there.  A label alone on its line belongs to the next line
        that holds anything."""
        self.data: list[tuple[int, str, list[str]]] = []
        top = DATA_BASE
        pending: list[str] = []
        by_line = {idx: name for name, idx in self.labels.items()}
        for idx, ins in enumerate(self.lines):
            name = by_line.get(idx)
            if name is not None:
                pending.append(name)
            if ins is None or ins[0] in DIRECTIVES:
                continue
            op, ops = ins
            if op in DATA_OPS:
                size = data_size(op, ops)
                if size is None:
                    raise SimError(f"size of {op} {ops} not known")
                for label in pending:
                    del self.labels[label]
                    self.symbols[label] = top
                if op not in ("ds", "defs"):
                    self.data.append((top, op, ops))
                top += size
            pending = []

    def _initialise_data(self) -> None:
        """What ``db`` and ``dw`` put in memory before the program runs."""
        for addr, op, ops in self.data:
            for item in ops:
                it = item.strip()
                if op in ("dw", "defw"):
                    v = self.eval(it) & 0xFFFF
                    self.mem[addr], self.mem[addr + 1] = v & 0xFF, v >> 8
                    addr += 2
                elif len(it) >= 2 and it[0] == it[-1] and it[0] in "'\"":
                    for ch in it[1:-1].replace(it[0] * 2, it[0]):
                        self.mem[addr] = ord(ch) & 0xFF
                        addr += 1
                else:
                    self.mem[addr] = self.eval(it) & 0xFF
                    addr += 1

    # ---- values -----------------------------------------------------------
    def eval(self, expr: str) -> int:
        """An operand expression: numbers, symbols, labels, + and -, and
        ``LOW``, ``HIGH`` of a term, as ``LOW(x)`` or M80's ``LOW x``."""
        s = re.sub(r"(?i)(?<![\w?@$.])(low|high)\s+([\w?@$.]+)", r"\1(\2)", expr.strip())
        if not s:
            raise SimError("empty expression")
        total = 0
        for sign, term in re.findall(r"([+-]?)\s*([^+-]+)", s.replace(" ", "")):
            v = self._term(term)
            total = total - v if sign == "-" else total + v
        return total

    def _term(self, term: str) -> int:
        t = term.strip()
        if len(t) == 3 and t[0] == t[2] == "'":
            return ord(t[1])
        v = parse_number(t)
        if v is not None:
            return v
        low = t.lower()
        if low in self.symbols:
            return self.symbols[low]
        if low in self.labels:
            return CODE_BASE + self.labels[low]
        m = re.match(r"^(low|high)\((.*)\)$", low)
        if m:
            v = self.eval(m.group(2))
            return v & 0xFF if m.group(1) == "low" else (v >> 8) & 0xFF
        raise SimError(f"unknown symbol {t}")

    def rp(self, name: str) -> int:
        if name == "af":
            return (self.r["a"] << 8) | self.f
        if name == "sp":
            return self.sp
        if name == "ix":
            return self.ix
        if name == "iy":
            return self.iy
        return (self.r[name[0]] << 8) | self.r[name[1]]

    def set_rp(self, name: str, v: int) -> None:
        v &= 0xFFFF
        if name == "af":
            self.r["a"], self.f = v >> 8, v & 0xFF
        elif name == "sp":
            self.sp = v
        elif name == "ix":
            self.ix = v
        elif name == "iy":
            self.iy = v
        else:
            self.r[name[0]], self.r[name[1]] = v >> 8, v & 0xFF

    def addr(self, operand: str) -> int | None:
        """The address of a memory operand, or None if it is not one."""
        o = operand.replace(" ", "").lower()
        if not (o.startswith("(") and o.endswith(")")):
            return None
        inner = o[1:-1]
        if inner in ("hl", "bc", "de"):
            return self.rp(inner)
        if inner == "sp":
            return self.sp
        m = re.match(r"^(ix|iy)([+-].*)?$", inner)
        if m:
            base = self.ix if m.group(1) == "ix" else self.iy
            d = self.eval(m.group(2).replace("+-", "-")) if m.group(2) else 0
            if not -128 <= d <= 127:
                raise SimError(f"displacement out of range in {operand}")
            return (base + d) & 0xFFFF
        return self.eval(operand.strip()[1:-1]) & 0xFFFF

    def get8(self, operand: str) -> int:
        o = operand.strip().lower()
        if o in R8:
            return self.r[o]
        a = self.addr(operand)
        if a is not None:
            return self.mem[a]
        v = self.eval(operand)
        if not -128 <= v <= 255:
            raise SimError(f"byte operand out of range: {operand}")
        return v & 0xFF

    def set8(self, operand: str, v: int) -> None:
        o = operand.strip().lower()
        if o in R8:
            self.r[o] = v & 0xFF
            return
        a = self.addr(operand)
        if a is None:
            raise SimError(f"cannot store to {operand}")
        self.mem[a] = v & 0xFF

    def read16(self, a: int) -> int:
        return self.mem[a & 0xFFFF] | (self.mem[(a + 1) & 0xFFFF] << 8)

    def write16(self, a: int, v: int) -> None:
        self.mem[a & 0xFFFF] = v & 0xFF
        self.mem[(a + 1) & 0xFFFF] = (v >> 8) & 0xFF

    def push(self, v: int) -> None:
        self.sp = (self.sp - 2) & 0xFFFF
        self.write16(self.sp, v)

    def pop(self) -> int:
        v = self.read16(self.sp)
        self.sp = (self.sp + 2) & 0xFFFF
        return v

    def target(self, label: str) -> int | None:
        low = label.strip().lower()
        if low in self.labels:
            return self.labels[low]
        v = parse_number(low)
        if v == 0:
            return None  # jp 0: warm boot, the end
        raise SimError(f"jump to unknown {label}")

    # ---- ALU -------------------------------------------------------------
    def add8(self, v: int, carry: int = 0) -> int:
        a = self.r["a"]
        r = a + v + carry
        f = szp(r) & ~FP
        if ((a & 0xF) + (v & 0xF) + carry) > 0xF:
            f |= FH
        if (~(a ^ v) & (a ^ r)) & 0x80:
            f |= FP
        if r > 0xFF:
            f |= FC
        self.f = f
        return r & 0xFF

    def sub8(self, v: int, carry: int = 0) -> int:
        a = self.r["a"]
        r = a - v - carry
        f = (szp(r) & ~FP) | FN
        if (a & 0xF) - (v & 0xF) - carry < 0:
            f |= FH
        if ((a ^ v) & (a ^ r)) & 0x80:
            f |= FP
        if r < 0:
            f |= FC
        self.f = f
        return r & 0xFF

    def logic(self, r: int, h: bool) -> None:
        self.f = szp(r) | (FH if h else 0)

    def inc8(self, v: int) -> int:
        r = (v + 1) & 0xFF
        f = (self.f & FC) | (szp(r) & ~FP)
        if (v & 0xF) == 0xF:
            f |= FH
        if v == 0x7F:
            f |= FP
        self.f = f
        return r

    def dec8(self, v: int) -> int:
        r = (v - 1) & 0xFF
        f = (self.f & FC) | (szp(r) & ~FP) | FN
        if (v & 0xF) == 0:
            f |= FH
        if v == 0x80:
            f |= FP
        self.f = f
        return r

    def cb(self, op: str, v: int) -> int:
        c = self.f & FC
        if op == "rlc":
            co = v >> 7
            r = ((v << 1) | co) & 0xFF
        elif op == "rrc":
            co = v & 1
            r = (v >> 1) | (co << 7)
        elif op == "rl":
            co = v >> 7
            r = ((v << 1) | c) & 0xFF
        elif op == "rr":
            co = v & 1
            r = (v >> 1) | (c << 7)
        elif op == "sla":
            co = v >> 7
            r = (v << 1) & 0xFF
        elif op == "sra":
            co = v & 1
            r = (v >> 1) | (v & 0x80)
        elif op == "srl":
            co = v & 1
            r = v >> 1
        elif op in ("sll", "sl1"):
            co = v >> 7
            r = ((v << 1) | 1) & 0xFF
        else:
            raise SimError(op)
        self.f = szp(r) | (FC if co else 0)
        return r

    # ---- execution ---------------------------------------------------------
    def run(self, max_steps: int = 20000) -> str:
        """Run from line 0: 'ret', 'halt', 'boot', 'end' or 'timeout'.

        The run starts as if called: a return address that is no line of the
        program is pushed, and a ``ret`` that pops it ends the run.  The
        data the text defines with ``db`` and ``dw`` is put in memory first."""
        self._initialise_data()
        self.push(SENTINEL)
        while True:
            if self.pc >= len(self.lines):
                return "end"
            ins = self.lines[self.pc]
            if ins is None:
                self.pc += 1
                continue
            self.steps += 1
            if self.steps > max_steps:
                return "timeout"
            op, ops = ins
            nxt = self.pc + 1
            res = self.step(op, ops, nxt)
            if isinstance(res, str):
                return res
            self.pc = res

    def jump(self, label: str) -> int | str:
        t = self.target(label)
        return "boot" if t is None else t

    def step(self, op: str, ops: list[str], nxt: int) -> int | str:  # noqa: C901
        n = len(ops)
        lo = [o.replace(" ", "").lower() for o in ops]
        if op in DATA_OPS:
            raise SimError("ran into data")
        if op in DIRECTIVES:
            return nxt
        if op == "nop" or op in ("di", "ei"):
            return nxt
        if op == "halt":
            return "halt"
        if op == "ld":
            d, s = lo
            if d in ("bc", "de", "hl", "sp", "ix", "iy"):
                if s in ("hl", "ix", "iy") and d == "sp":
                    self.sp = self.rp(s)
                elif s.startswith("(") and s.endswith(")"):
                    self.set_rp(d, self.read16(self.eval(ops[1].strip()[1:-1])))
                else:
                    self.set_rp(d, self.eval(ops[1]))
                return nxt
            if s in ("bc", "de", "hl", "sp", "ix", "iy"):
                a = self.eval(ops[0].strip()[1:-1])
                self.write16(a, self.rp(s))
                return nxt
            self.set8(ops[0], self.get8(ops[1]))
            return nxt
        if op == "push":
            self.push(self.rp(lo[0]))
            return nxt
        if op == "pop":
            self.set_rp(lo[0], self.pop())
            return nxt
        if op == "ex":
            if set(lo) == {"de", "hl"}:
                de, hl = self.rp("de"), self.rp("hl")
                self.set_rp("de", hl)
                self.set_rp("hl", de)
                return nxt
            if lo[0] == "(sp)":
                v = self.read16(self.sp)
                self.write16(self.sp, self.rp(lo[1]))
                self.set_rp(lo[1], v)
                return nxt
            raise SimError(f"ex {ops}")
        if op in ("add", "adc", "sub", "sbc", "and", "or", "xor", "cp"):
            if n == 2 and lo[0] in ("hl", "ix", "iy"):
                hl = self.rp(lo[0])
                v = self.rp(lo[1])
                if op == "add":
                    r = hl + v
                    f = self.f & (FS | FZ | FP)
                    if ((hl & 0xFFF) + (v & 0xFFF)) > 0xFFF:
                        f |= FH
                    if r > 0xFFFF:
                        f |= FC
                    self.f = f
                elif op == "adc":
                    c = self.f & FC
                    r = hl + v + c
                    f = 0
                    if (r & 0xFFFF) == 0:
                        f |= FZ
                    if r & 0x8000:
                        f |= FS
                    if ((hl & 0xFFF) + (v & 0xFFF) + c) > 0xFFF:
                        f |= FH
                    if (~(hl ^ v) & (hl ^ r)) & 0x8000:
                        f |= FP
                    if r > 0xFFFF:
                        f |= FC
                    self.f = f
                elif op == "sbc":
                    c = self.f & FC
                    r = hl - v - c
                    f = FN
                    if (r & 0xFFFF) == 0:
                        f |= FZ
                    if r & 0x8000:
                        f |= FS
                    if (hl & 0xFFF) - (v & 0xFFF) - c < 0:
                        f |= FH
                    if ((hl ^ v) & (hl ^ r)) & 0x8000:
                        f |= FP
                    if r < 0:
                        f |= FC
                    self.f = f
                else:
                    raise SimError(f"{op} {ops}")
                self.set_rp(lo[0], r & 0xFFFF)
                return nxt
            src = ops[1] if n == 2 else ops[0]
            if n == 2 and lo[0] != "a":
                raise SimError(f"{op} {ops}")
            v = self.get8(src)
            c = self.f & FC
            if op == "add":
                self.r["a"] = self.add8(v)
            elif op == "adc":
                self.r["a"] = self.add8(v, c)
            elif op == "sub":
                self.r["a"] = self.sub8(v)
            elif op == "sbc":
                self.r["a"] = self.sub8(v, c)
            elif op == "cp":
                self.sub8(v)
            elif op == "and":
                self.r["a"] &= v
                self.logic(self.r["a"], True)
            elif op == "or":
                self.r["a"] |= v
                self.logic(self.r["a"], False)
            else:
                self.r["a"] ^= v
                self.logic(self.r["a"], False)
            return nxt
        if op in ("inc", "dec"):
            o = lo[0]
            if o in ("bc", "de", "hl", "sp", "ix", "iy"):
                self.set_rp(o, self.rp(o) + (1 if op == "inc" else -1))
                return nxt
            v = self.get8(ops[0])
            self.set8(ops[0], self.inc8(v) if op == "inc" else self.dec8(v))
            return nxt
        if op == "cpl":
            self.r["a"] ^= 0xFF
            self.f |= FH | FN
            return nxt
        if op == "neg":
            a = self.r["a"]
            self.r["a"] = 0
            self.r["a"] = self.sub8(a)
            return nxt
        if op == "ccf":
            c = self.f & FC
            self.f = (self.f & (FS | FZ | FP)) | (FH if c else 0) | (0 if c else FC)
            return nxt
        if op == "scf":
            self.f = (self.f & (FS | FZ | FP)) | FC
            return nxt
        if op == "daa":
            a = self.r["a"]
            c = self.f & FC
            h = self.f & FH
            nflag = self.f & FN
            corr = 0
            carry = c
            if h or (a & 0xF) > 9:
                corr |= 0x06
            if c or a > 0x99:
                corr |= 0x60
                carry = FC
            if nflag:
                r = (a - corr) & 0xFF
                hh = FH if (h and (a & 0xF) < 6) else 0
            else:
                r = (a + corr) & 0xFF
                hh = FH if (a & 0xF) > 9 else 0
            self.r["a"] = r
            self.f = szp(r) | hh | nflag | carry
            return nxt
        if op in ("rlca", "rrca", "rla", "rra"):
            a = self.r["a"]
            c = self.f & FC
            if op == "rlca":
                co = a >> 7
                a = ((a << 1) | co) & 0xFF
            elif op == "rrca":
                co = a & 1
                a = (a >> 1) | (co << 7)
            elif op == "rla":
                co = a >> 7
                a = ((a << 1) | c) & 0xFF
            else:
                co = a & 1
                a = (a >> 1) | (c << 7)
            self.r["a"] = a
            self.f = (self.f & (FS | FZ | FP)) | (FC if co else 0)
            return nxt
        if op in ("rlc", "rrc", "rl", "rr", "sla", "sra", "srl", "sll", "sl1"):
            self.set8(ops[0], self.cb(op, self.get8(ops[0])))
            return nxt
        if op in ("bit", "set", "res"):
            b = int(ops[0])
            v = self.get8(ops[1])
            if op == "bit":
                zero = not (v >> b) & 1
                f = (self.f & FC) | FH
                if zero:
                    f |= FZ | FP
                if b == 7 and not zero:
                    f |= FS
                self.f = f
            elif op == "set":
                self.set8(ops[1], v | (1 << b))
            else:
                self.set8(ops[1], v & ~(1 << b))
            return nxt
        if op in ("jp", "jr"):
            if n == 1:
                if lo[0] in ("(hl)", "(ix)", "(iy)"):
                    dest = self.rp(lo[0][1:-1]) - CODE_BASE
                    if not 0 <= dest < len(self.lines):
                        raise SimError("indirect jump out of the program")
                    return dest
                return self.jump(ops[0])
            if COND[lo[0]](self.f):
                return self.jump(ops[1])
            return nxt
        if op == "djnz":
            self.r["b"] = (self.r["b"] - 1) & 0xFF
            if self.r["b"]:
                return self.jump(ops[0])
            return nxt
        if op == "call":
            if n == 2:
                if not COND[lo[0]](self.f):
                    return nxt
                label = ops[1]
            else:
                label = ops[0]
            t = self.target(label)
            if t is None:
                raise SimError("call 0")
            self.push(CODE_BASE + nxt)
            return t
        if op == "ret":
            if n == 1 and not COND[lo[0]](self.f):
                return nxt
            back = self.pop()
            if back == SENTINEL:
                return "ret"
            if not 0 <= back - CODE_BASE < len(self.lines):
                raise SimError("return to an address outside the program")
            return back - CODE_BASE
        if op in ("ldir", "lddr", "ldi", "ldd"):
            while True:
                self.mem[self.rp("de")] = self.mem[self.rp("hl")]
                step = 1 if op in ("ldir", "ldi") else -1
                self.set_rp("hl", self.rp("hl") + step)
                self.set_rp("de", self.rp("de") + step)
                self.set_rp("bc", self.rp("bc") - 1)
                if op in ("ldi", "ldd") or self.rp("bc") == 0:
                    break
            self.f = (self.f & (FS | FZ | FC)) | (FP if self.rp("bc") else 0)
            return nxt
        raise SimError(f"cannot run {op} {ops}")

    def state(self, mem_ranges: list[tuple[int, int]]) -> dict:
        """Registers, flags (bits 3 and 5 masked) and the given memory."""
        st = {k: v for k, v in self.r.items()}
        st["f"] = self.f & FLAG_MASK
        st["ix"] = self.ix
        st["iy"] = self.iy
        st["sp"] = self.sp
        for lo, hi in mem_ranges:
            st[f"mem{lo:04x}"] = bytes(self.mem[lo:hi])
        return st
