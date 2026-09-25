"""Differential testing of the peephole optimizer.

    python3 tests/peepfuzz.py [--seeds N] [--first S] [--states K] [-v]

Each seed is a random Z80 program: straight-line code, forward branches,
counted loops, calls to subroutines of its own and to the multiply and
subtract routines uplm80 links in, all built around the instruction
sequences the optimizer rewrites, with random operands and random code
before and after them.  The program and its optimized version are run on
:mod:`tests.z80sim` from the same random register, flag and memory states,
and must end in the same state: every register, every flag (bits 3 and 5
aside) and all of memory outside the stack below SP.  The program ends in
``ret`` (or ``jp 0``), where the optimizer has to assume everything is
live, so a rewrite that changed any register or flag some path could still
read shows up as a difference.  Every instruction of the optimized program
must also be one the Z80 has.

A failing seed prints both programs and the states that differ.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from upeepz80 import PeepholeOptimizer  # noqa: E402
from upeepz80.z80 import UNKNOWN, effect, strip_comment  # noqa: E402
from z80sim import FLAG_MASK, Machine, SimError  # noqa: E402

VARS = {f"V{i}": 0x8000 + 2 * i for i in range(8)}
SYMBOLS = dict(VARS, FRAME=0x9000)
IX_BASE = 0x9080
STACK_TOP = 0xF000
STACK_LOW = 0xE000

R8 = ["a", "b", "c", "d", "e", "h", "l"]
R16 = ["bc", "de", "hl"]
ALU = ["add a,", "adc a,", "sub ", "sbc a,", "and ", "or ", "xor ", "cp "]
CB = ["rlc", "rrc", "rl", "rr", "sla", "sra", "srl"]
CC = ["z", "nz", "c", "nc", "pe", "po", "p", "m"]
JRCC = ["z", "nz", "c", "nc"]

RUNTIME = """\
??mul16:
\tld\tb,h
\tld\tc,l
\tld\thl,0
??mul16l:
\tld\ta,e
\tor\td
\tret\tz
\tsrl\td
\trr\te
\tjp\tnc,??mul16s
\tadd\thl,bc
??mul16s:
\tsla\tc
\trl\tb
\tjp\t??mul16l
??subde:
\tor\ta
\tsbc\thl,de
\tret
"""


class Gen:
    """One random program."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.labels = 0
        self.pending: list[str] = []

    def label(self) -> str:
        self.labels += 1
        return f"??L{self.labels}"

    def var(self) -> str:
        return self.rng.choice(list(VARS))

    def byte(self) -> str:
        r = self.rng.random()
        if r < 0.2:
            return "0"
        if r < 0.3:
            return "0ffh"
        if r < 0.4:
            return "1"
        return str(self.rng.randrange(256))

    def word(self) -> str:
        r = self.rng.random()
        if r < 0.15:
            return "0"
        if r < 0.25:
            return "1"
        if r < 0.35:
            return "0ffffh"
        if r < 0.5:
            return self.var()
        if r < 0.6:
            return str(self.rng.randrange(256, 700))
        return str(self.rng.randrange(0x10000))

    def idx(self) -> str:
        d = self.rng.randrange(-8, 8)
        if d < 0:
            return self.rng.choice([f"(ix{d})", f"(ix+{d})"])
        return f"(ix+{d})"

    def src8(self) -> str:
        r = self.rng.random()
        if r < 0.55:
            return self.rng.choice(R8)
        if r < 0.75:
            return self.byte()
        if r < 0.9:
            return "(hl)"
        return self.idx()

    # ---- single instructions ----------------------------------------------
    def instr(self, allow_b: bool = True) -> list[str]:
        g = self.rng
        k = g.randrange(30)
        regs = R8 if allow_b else [r for r in R8 if r != "b"]
        pairs = R16 if allow_b else ["de", "hl"]
        if k == 0:
            return [f"ld {g.choice(regs)},{g.choice(R8)}"]
        if k == 1:
            return [f"ld {g.choice(regs)},{self.byte()}"]
        if k == 2:
            return [f"ld {g.choice(regs)},(hl)"]
        if k == 3:
            return [f"ld (hl),{g.choice(R8)}"]
        if k == 4:
            return [f"ld a,({self.var()})"]
        if k == 5:
            return [f"ld ({self.var()}),a"]
        if k == 6:
            return [f"ld {g.choice(pairs)},({self.var()})"]
        if k == 7:
            return [f"ld ({self.var()}),{g.choice(R16)}"]
        if k == 8:
            return [f"ld {g.choice(pairs)},{self.word()}"]
        if k == 9:
            return [f"{g.choice(ALU)}{self.src8()}"]
        if k == 10:
            return [f"{g.choice(['inc', 'dec'])} {g.choice(regs + ['(hl)', self.idx()])}"]
        if k == 11:
            return [f"{g.choice(['inc', 'dec'])} {g.choice(pairs)}"]
        if k == 12:
            return [f"add hl,{g.choice(R16)}"]
        if k == 13:
            return [f"{g.choice(['adc', 'sbc'])} hl,{g.choice(R16)}"]
        if k == 14:
            return ["ex de,hl"]
        if k == 15:
            return [g.choice(["cpl", "neg", "ccf", "scf", "daa", "rla", "rra", "rlca", "rrca", "nop"])]
        if k == 16:
            return [f"{g.choice(CB)} {g.choice(regs + ['(hl)'])}"]
        if k == 17:
            return [f"bit {g.randrange(8)},{g.choice(R8 + ['(hl)'])}"]
        if k == 18:
            return [f"{g.choice(['set', 'res'])} {g.randrange(8)},{g.choice(regs + ['(hl)'])}"]
        if k == 19:
            return [f"ld a,{self.idx()}"]
        if k == 20:
            return [f"ld {self.idx()},{g.choice([g.choice(R8), self.byte()])}"]
        if k == 21:
            p, q = g.choice(R16 + ["af"]), g.choice(pairs + ["af"])
            return [f"push {p}"] + self.instr(allow_b) + [f"pop {q}"]
        if k == 22:
            return [f"ld hl,{self.var()}"]
        if k == 23:
            # read the flags as data
            return ["push af", f"pop {g.choice(pairs)}"]
        if k == 24:
            return [f"adc a,{self.byte()}"]
        if k == 25:
            return ["sbc a,a"]
        if k == 26:
            return [f"xor a"]
        if k == 27:
            return [f"cp {g.choice(R8)}"]
        if k == 28:
            return [f"ld {g.choice(['hl', 'de'])},{self.var()}", f"ld a,({g.choice(['hl', 'de'])})"]
        return [f"ld a,{self.byte()}"]

    # ---- the shapes the optimizer rewrites ----------------------------------
    def seed(self, allow_b: bool = True) -> list[str]:  # noqa: C901
        g = self.rng
        v, w = self.var(), self.var()
        r = g.choice([x for x in R8 if allow_b or x != "b"])
        shapes = [
            lambda: ["push " + (p := g.choice(R16 + ["af"])), "pop " + p],
            lambda: [f"ld a,{(x := g.choice(R8 + ['(hl)', f'({v})', self.idx()]))}", f"ld {x},a"],
            lambda: ["ld a,0"],
            lambda: ["cp 0"],
            lambda: [f"ld {(x := g.choice(['a', 'e', 'l', 'h', r]))},{(y := g.choice(['(hl)', 'b', f'({v})' if x == 'a' else 'c', self.idx()]))}", f"ld {x},{y}"],
            lambda: [f"ld {r},{r}"],
            lambda: ["inc a", "dec a"],
            lambda: ["dec a", "inc a"],
            lambda: [f"inc {(p := g.choice(R16 if allow_b else ['de', 'hl']))}", f"dec {p}"],
            lambda: [f"dec {(p := g.choice(R16 if allow_b else ['de', 'hl']))}", f"inc {p}"],
            lambda: ["or a", "or a"],
            lambda: ["and a", "and a"],
            lambda: ["xor a", "xor a"],
            lambda: ["ex de,hl", "ex de,hl"],
            lambda: ["push hl", "ex (sp),hl", "ex (sp),hl", "pop hl"],
            lambda: ["ccf", "ccf"],
            lambda: ["cpl", "cpl"],
            lambda: [f"push {(p := g.choice(R16))}", f"pop {g.choice([q for q in R16 if q != p and (allow_b or q != 'bc')])}"],
            lambda: ["ccf", "scf"],
            lambda: ["ld a,(hl)", f"ld {g.choice([x for x in 'bcde' if allow_b or x != 'b'])},a"],
            lambda: [f"ld {(x := g.choice([y for y in 'bcdehl' if allow_b or y != 'b']))},a", f"ld a,{x}"],
            lambda: [f"ld ({v}),hl", f"ld hl,({v})"],
            lambda: [f"ld ({v}),a", f"ld a,({v})"],
            lambda: ["and 0ffh"],
            lambda: ["or 0"],
            lambda: ["xor 0"],
            lambda: ["push hl", "ex de,hl", "pop hl"],
            lambda: ["ld h,0", "ld d,h", "ld e,l"],
            lambda: ["ld l,a", "ld h,0", f"sub {g.choice(['l', 'h', '(hl)', 'b', self.byte()])}"],
            lambda: ["ld l,a", "ld h,0", f"cp {g.choice(['l', 'h', '(hl)', 'c', self.byte()])}"],
            lambda: ["ld l,a", "ld h,0", "ld l,a", "ld h,0"],
            lambda: ["ld l,a", "ld h,0", "push hl", "ld l,a"] + self.instr(allow_b) + ["pop " + g.choice(["hl", "de"])],
            lambda: ["ld hl,0ffffh", "ld a,l", "or h"],
            lambda: ["ld hl,1", "ld a,l", "or h"],
            lambda: [f"ld hl,1", f"ld {g.choice([x for x in 'c' if allow_b] + ['c'])},l"],
            lambda: ["ld hl,0", "ld a,l", "or h"],
            lambda: ["push hl", f"ld ({v}),hl", "pop hl"],
            lambda: ["push af", f"ld ({v}),a", "pop af"],
            lambda: ["ld a,l", "ld h,0", f"ld {g.choice([f'({v})', '(hl)', self.idx()])},a"],
            lambda: ["ld l,a", "ld h,0", f"ld {g.choice([f'({v})', '(hl)', self.idx()])},a"],
            lambda: ["ld a,l", "ld h,0", "or h"],
            lambda: ["ld h,0", "or h"],
            lambda: [f"ld de,{g.randrange(1, 5)}", "add hl,de"],
            lambda: [f"ld a,({v})", "inc a", f"ld ({v}),a"],
            lambda: [f"ld a,({v})", "dec a", f"ld ({v}),a"],
            lambda: [f"ld a,{(x := self.idx())}", g.choice(["inc a", "dec a"]), f"ld {x},a"],
            lambda: ["ld a,(hl)", g.choice(["inc a", "dec a"]), "ld (hl),a"],
            lambda: [f"ld a,({v})", "cpl", "cpl", "inc a", "push af", f"ld ({v}),a", "pop af"],
            lambda: ["or a", "ld a,h", "rra", "ld h,a", "ld a,l", "rra", "ld l,a"],
            lambda: ["push hl", f"ld hl,({v})", "ex de,hl", "pop hl"],
            lambda: [f"ld hl,{g.choice([self.byte(), self.word(), '299', '-1'])}", f"ld {g.choice([x for x in 'abcde' if allow_b or x != 'b'])},l"],
            lambda: ["push hl"] + self.instr(allow_b) + ["pop hl", "push hl", f"ld hl,{self.word()}"] + self.instr(allow_b) + ["pop hl"],
            lambda: ["ld hl,0", "ld a,l", f"ld {g.choice([f'({v})', '(hl)', self.idx(), '(de)'])},a"],
            lambda: [f"ld hl,({v})", "push hl", f"ld hl,({w})", "ex de,hl", "pop hl"],
            lambda: [f"ld hl,{g.choice(['9', '0', '16', '299'])}", "ld a,l", f"ld ({v}),a", f"ld ({w}),hl"],
            lambda: [f"ld hl,{self.word()}", "ld a,l", "ld l,a", "ld h,0", f"ld ({w}),hl"],
            lambda: [f"ld a,({v})", f"{g.choice(['cp ' + self.byte(), 'or a'])}", f"jp {g.choice(['z', 'nz'])},{self.forward()}", f"ld a,({v})"],
            lambda: [f"jp {(c := g.choice(['z', 'nz', 'c', 'nc']))},{(t := self.forward())}", f"jp {t}"],
        ]
        if allow_b:
            shapes += [
                lambda: [f"ld de,{g.choice([2, 4, 8, 16, 32, 64, 128, 3])}", "call ??mul16"],
                lambda: ["ld de,0", "call ??subde"],
                lambda: [f"ld de,{self.word()}", "call ??subde"],
            ]
        return g.choice(shapes)()

    def plain(self, allow_b: bool = True) -> list[str]:
        """An instruction that does not touch the stack."""
        while True:
            ins = self.instr(allow_b)
            if not any(i.startswith(("push", "pop")) for i in ins):
                return ins

    def flag_reader(self) -> str:
        g = self.rng
        return g.choice([f"jp {g.choice(CC)},{self.forward()}", "adc a,0", "sbc a,a", "rla", "daa",
                         f"jr {g.choice(JRCC)},{self.forward()}"])

    def kill(self, allow_b: bool = True) -> list[str]:
        """Overwrite some registers and the flags, in a random order."""
        g = self.rng
        pool = [f"ld a,{self.byte()}", f"ld de,{self.word()}", f"ld hl,{self.word()}", f"ld c,{self.byte()}",
                g.choice(["cp b", "cp c", "xor a", "or a", "and 0fh"])]
        if allow_b:
            pool.append(f"ld bc,{self.word()}")
        g.shuffle(pool)
        return pool[:g.randrange(2, len(pool) + 1)]

    def forward(self) -> str:
        lab = self.label()
        self.pending.append(lab)
        return lab

    def chunk(self, allow_b: bool = True, depth: int = 0) -> list[str]:
        g = self.rng
        k = g.random()
        if k < 0.45:
            out = self.seed(allow_b)
            r = g.random()
            if r < 0.3:
                # overwrite what the rewrite may change, so that it can happen
                out += self.kill(allow_b)
            elif r < 0.4:
                # carry the flags and A through the stack, and read them
                out += ["push af"] + self.plain(allow_b) + ["pop af", self.flag_reader()]
            elif r < 0.5 and allow_b:
                # a conditional call: the rewrite has to hold whether it is made or not
                out += [f"call {g.choice(CC)},{g.choice(['??S1', '??S2'])}", self.flag_reader()]
            return out
        if k < 0.75:
            return self.instr(allow_b)
        if k < 0.82:
            op = g.choice(["jp", "jr"])
            cc = g.choice(CC if op == "jp" else JRCC)
            return [f"{op} {cc},{self.forward()}"]
        if k < 0.85:
            # an unconditional jump, and what follows it is reached only by label
            lab = self.forward()
            return [f"jp {lab}", f"{self.label()}:"]
        if k < 0.88:
            return [f"ret {g.choice(CC)}"]
        if k < 0.93 and allow_b and depth == 0:
            body: list[str] = []
            for _ in range(g.randrange(1, 4)):
                body += self.chunk(allow_b=False, depth=1)
            top = self.label()
            close = g.choice([["dec b", f"jp nz,{top}"], ["dec b", f"jr nz,{top}"], [f"djnz {top}"]])
            return [f"ld b,{g.randrange(1, 5)}", f"{top}:"] + body + close
        if k < 0.97 and allow_b:
            call = [g.choice(["call ??S1", f"call {g.choice(CC)},??S1", "call ??S2", "call ??S2"])]
            r = g.random()
            if r < 0.3:
                # read what the routine leaves in the flags
                call.append(g.choice([f"jp {g.choice(CC)},{self.forward()}", "adc a,0", "sbc a,a",
                                      "push af", "rla"]))
                if call[-1] == "push af":
                    call += self.instr(allow_b) + [g.choice(["pop af", "pop hl", "pop de"])]
            elif r < 0.4:
                # a routine entered by a jump, returning to an address pushed
                back = self.label()
                call = [f"ld hl,{back}", "push hl", f"ld hl,{self.word()}", "jp ??S3", f"{back}:"]
            return call
        if self.pending:
            return [f"{self.pending.pop(g.randrange(len(self.pending)))}:"]
        return self.instr(allow_b)

    def program(self, size: int) -> str:
        g = self.rng
        out: list[str] = []
        for _ in range(size):
            if self.pending and g.random() < 0.15:
                out.append(f"{self.pending.pop(g.randrange(len(self.pending)))}:")
            out += self.chunk()
        for lab in self.pending:
            out.append(f"{lab}:")
        self.pending = []
        # a thread of jumps, and a label after an unconditional return
        if g.random() < 0.5:
            a, b = self.label(), self.label()
            out = [f"jp {g.choice(['z', 'nz', 'c'])},{a}"] + out + [f"jr {b}", f"{a}:", f"jp {b}", f"{b}:"]
        out += self.instr() + [g.choice(["ret", "ret", "jp 0"])]
        subs: list[str] = []
        for name in ("??S1", "??S2", "??S3"):
            body = []
            if g.random() < 0.3:
                # read the flags or registers the caller left
                body += [g.choice(["adc a,1", "rla", f"jr nc,{self.forward()}", "push af", "ld (V0),hl",
                                   "sbc hl,de", "ld a,l"])]
                if body[-1] == "push af":
                    body += self.instr() + ["pop af"]
            for _ in range(g.randrange(1, 5)):
                body += self.seed(allow_b=True) if g.random() < 0.6 else self.instr()
            if g.random() < 0.3:
                body += [g.choice(["call ??S2", "call ??mul16"]) if name == "??S1" else "nop", "ret"]
            if g.random() < 0.2:
                # a jump made by pushing an address and returning to it
                there = self.label()
                body += [f"ld hl,{there}", "push hl", f"ld hl,{self.word()}",
                         g.choice(["ret", f"ret {g.choice(CC)}"])] + self.plain() + \
                    [f"{there}:", self.flag_reader()]
            if g.random() < 0.2:
                # take the return address off the stack and put it back
                # (then overwrite the copy: code addresses differ once optimized)
                body += g.choice([["pop hl", "push hl", f"ld hl,{self.word()}"],
                                  ["ex (sp),hl", "ex (sp),hl"],
                                  ["pop de", "push de", f"ld de,{self.word()}"]])
            # leave something in the flags for the caller
            body += [g.choice(["cp b", "or a", "xor a", "inc a", "scf", "and 0fh", "nop"])]
            body += [f"{lab}:" for lab in self.pending]
            self.pending = []
            # ??S2 may be exported, and so entered from outside as well
            colon = "::" if name == "??S2" and g.random() < 0.3 else ":"
            subs += [f"{name}{colon}"] + body + ["ret"]
        text = []
        for line in out + subs:
            text.append(line if line.endswith(":") else "\t" + line)
        return "\n".join(text) + "\n" + RUNTIME


def run(src: str, init: dict) -> tuple[str, Machine]:
    m = Machine(src, SYMBOLS)
    for k in ("a", "b", "c", "d", "e", "h", "l"):
        m.r[k] = init[k]
    m.f = init["f"]
    m.ix = IX_BASE
    m.iy = init["iy"]
    m.sp = STACK_TOP
    m.mem[:] = init["mem"]
    guard = _Guard(m)
    how = guard.run()
    return how, m


class _Guard:
    """Refuse memory operands that reach the stack area: what lies below SP
    differs between a program and its optimized version (a push the
    optimizer removed leaves nothing there), and reading it is no test."""

    def __init__(self, m: Machine):
        self.m = m
        orig = m.addr

        def addr(operand: str) -> int | None:
            a = orig(operand)
            if a is not None and (STACK_LOW - 2 <= a < STACK_TOP + 16):
                raise SimError("memory operand in the stack area")
            return a

        m.addr = addr  # type: ignore[method-assign]

    def run(self) -> str:
        return self.m.run(max_steps=5000)


def random_state(rng: random.Random) -> dict:
    mem = bytearray(rng.randbytes(0x10000))
    st = {k: rng.randrange(256) for k in ("a", "b", "c", "d", "e")}
    # HL mostly addresses the variables, so that (hl) is interesting
    hl = rng.choice([0x8000 + rng.randrange(16), rng.randrange(0x10000)])
    if STACK_LOW - 16 <= hl < STACK_TOP + 32:
        hl = 0x8000
    st["h"], st["l"] = hl >> 8, hl & 0xFF
    st["f"] = rng.randrange(256)
    st["iy"] = rng.randrange(0x10000)
    st["mem"] = mem
    return st


def compare(a: Machine, b: Machine) -> list[str]:
    diffs = []
    for k in ("a", "b", "c", "d", "e", "h", "l"):
        if a.r[k] != b.r[k]:
            diffs.append(f"{k}: {a.r[k]:02X} vs {b.r[k]:02X}")
    if (a.f ^ b.f) & FLAG_MASK:
        diffs.append(f"f: {a.f & FLAG_MASK:08b} vs {b.f & FLAG_MASK:08b} (SZ-H-PNC)")
    for k in ("ix", "iy", "sp"):
        if getattr(a, k) != getattr(b, k):
            diffs.append(f"{k}: {getattr(a, k):04X} vs {getattr(b, k):04X}")
    low = min(a.sp, b.sp)
    if a.mem[:STACK_LOW] != b.mem[:STACK_LOW] or a.mem[low:] != b.mem[low:]:
        for addr in list(range(STACK_LOW)) + list(range(low, 0x10000)):
            if a.mem[addr] != b.mem[addr]:
                diffs.append(f"mem[{addr:04X}]: {a.mem[addr]:02X} vs {b.mem[addr]:02X}")
                if len(diffs) > 12:
                    break
    return diffs


def invalid_instructions(asm: str) -> list[str]:
    bad = []
    for line in asm.split("\n"):
        text = strip_comment(line)
        body = text
        if text and not text[0].isspace():
            if ":" not in text:
                continue
            body = text.split(":", 1)[1].removeprefix(":")
        body = body.strip()
        if not body:
            continue
        parts = body.split(None, 1)
        if effect(parts[0], parts[1] if len(parts) > 1 else "") is UNKNOWN:
            bad.append(line.strip())
    return bad


def check(seed: int, size: int = 12, states: int = 6) -> str | None:
    """None if seed ``seed`` behaves the same optimized, else a report."""
    rng = random.Random(seed)
    src = Gen(rng).program(size)
    opt = PeepholeOptimizer().optimize(src)
    bad = invalid_instructions(opt)
    if bad:
        return f"seed {seed}: not Z80 instructions: {bad}\n--- original\n{src}\n--- optimized\n{opt}"
    # An exported routine is also run as another module would: called, and
    # everything it leaves compared.  (The call is not part of what is
    # optimized.)
    entries = [""] + (["\tcall ??S2\n\tjp 0\n"] if "??S2::" in src else [])
    ran = 0
    for _ in range(states):
        init = random_state(rng)
        for entry in entries:
            try:
                how0, m0 = run(entry + src, init)
            except SimError:
                continue
            if how0 in ("timeout", "end"):
                continue
            ran += 1
            try:
                how1, m1 = run(entry + opt, init)
            except SimError as exc:
                return f"seed {seed}: optimized program failed: {exc}\n--- original\n{src}\n--- optimized\n{opt}"
            diffs = compare(m0, m1)
            if how0 != how1:
                diffs.insert(0, f"ended by {how0} vs {how1}")
            if diffs:
                return (f"seed {seed}{' entered by ' + entry.split(chr(10))[0].strip() if entry else ''}: " +
                        "; ".join(diffs[:8]) + f"\n--- original\n{src}\n--- optimized\n{opt}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("-v", action="store_true", help="print every failing program")
    args = ap.parse_args()
    fails = 0
    for seed in range(args.first, args.first + args.seeds):
        rep = check(seed, args.size, args.states)
        if rep:
            fails += 1
            print(rep if args.v or fails <= 3 else rep.split("\n", 1)[0])
            sys.stdout.flush()
    print(f"{fails} of {args.seeds} programs differ")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
