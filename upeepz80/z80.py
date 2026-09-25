"""
What a Z80 instruction reads, writes, where it can go next, and its size.

The peephole optimizer needs three facts about an instruction that pattern
text alone does not give it:

* the registers and flags it reads and writes, so that a rewrite which
  changes what a register or flag holds afterwards can first prove that
  nothing reads the old value (:func:`effect`);
* where control goes after it - on, to a label, both, into a routine that
  comes back, back to a caller, or somewhere the optimizer cannot follow
  (``jp (hl)``, an instruction it does not know);
* its length in bytes, so that a relative jump is only made where its
  displacement is known to fit.

Resources are named by strings: the eight-bit registers ``a b c d e h l``,
the halves of IX and IY (``ixh ixl iyh iyl``; the undocumented instructions
that name them are not recognised), ``sp``, ``i`` and ``r``, and the six
flags ``fs fz fh fp fn fc`` (sign, zero, half carry, parity/overflow,
subtract, carry).  The undocumented flag bits 3 and 5 are not modelled.

Every set here errs on the safe side for the optimizer.  A read set may
include more than the instruction really reads - that only keeps a value
alive longer.  A write set never includes a register or flag the instruction
may leave alone, since a write is what lets the optimizer call the old value
dead.  An instruction this module does not recognise reads everything, and
its size is unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REGS8 = frozenset("abcdehl")
FLAGS = frozenset({"fs", "fz", "fh", "fp", "fn", "fc"})
IX = frozenset({"ixh", "ixl"})
IY = frozenset({"iyh", "iyl"})
ALL = REGS8 | FLAGS | IX | IY | frozenset({"sp", "i", "r"})

# S, Z, H, P/V and N: what inc/dec and most other flag-setters without C write.
FLAGS_NO_C = FLAGS - {"fc"}

PAIRS: dict[str, frozenset[str]] = {
    "bc": frozenset("bc"),
    "de": frozenset("de"),
    "hl": frozenset("hl"),
    "sp": frozenset({"sp"}),
    "af": frozenset("a") | FLAGS,
    "ix": IX,
    "iy": IY,
}

CONDITIONS: dict[str, frozenset[str]] = {
    "z": frozenset({"fz"}), "nz": frozenset({"fz"}),
    "c": frozenset({"fc"}), "nc": frozenset({"fc"}),
    "pe": frozenset({"fp"}), "po": frozenset({"fp"}),
    "p": frozenset({"fs"}), "m": frozenset({"fs"}),
}
JR_CONDITIONS = frozenset({"z", "nz", "c", "nc"})

# Directives.  Those that emit nothing and do not move the location counter
# are transparent; data directives are data, and anything that moves the
# location counter or ends the section ends what can be measured.
TRANSPARENT = frozenset({
    "public", "global", "extrn", "extern", "ext", "entry", "name", "title",
    "subttl", "page", "eject", "list", "nolist", ".list", ".xlist", ".z80",
    ".8080", "radix", ".radix", "equ", "defl", ".printx",
})
DATA = frozenset({"db", "defb", "dw", "defw", "ds", "defs", "defm", "dc", "byte", "word"})
BARRIERS = frozenset({
    "org", "cseg", "dseg", "aseg", "common", "end", ".phase", ".dephase",
    "if", "ifdef", "ifndef", "else", "endif", "macro", "endm", "rept", "irp",
    "irpc", "exitm", "include", "maclib", "cond", "endc", ".comment",
})


@dataclass(frozen=True)
class Effect:
    """What one instruction does, as far as the optimizer is concerned."""

    reads: frozenset[str]
    writes: frozenset[str]
    # "next": falls through; "jump": goes to ``target`` only; "branch": to
    # ``target`` or on; "call": to ``target`` (None for rst), which returns
    # to the next line; "return": back to the caller; "stop": somewhere the
    # optimizer cannot know (jp (hl), reti, halt, an instruction it does not
    # recognise - these read ALL); "data": not an instruction at all (reads
    # ALL, as far as control running into it goes).  For "call" and
    # "return", ``reads`` is only the condition's flag: what the callee or
    # the caller reads is a matter for whoever follows them.
    flow: str
    target: str | None = None
    size: int | None = None
    # ``ex de,hl``: the contents of DE and HL change places.
    swap_de_hl: bool = False
    # A conditional jump, call or return (or djnz): it may also go on.
    cond: bool = False
    # What it does to SP: +1 a push, -1 a pop (of ``pair``), 0 nothing the
    # optimizer need follow (a call and its return balance), None anything
    # else (ld sp, inc sp, dec sp).
    stack: int | None = 0
    pair: str | None = None


UNKNOWN = Effect(ALL, frozenset(), "stop", None, None)


def strip_comment(text: str) -> str:
    """``text`` without a trailing ``;`` comment, leaving quoted strings alone."""
    quote = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == quote:
                if i + 1 < n and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif ch == ";":
            return text[:i].rstrip()
        elif ch in "'\"":
            # The prime of af' is not a quote.
            if not (ch == "'" and text[max(0, i - 2):i].lower() == "af"):
                quote = ch
        i += 1
    return text.rstrip()


def split_operands(text: str) -> list[str]:
    """Split an operand field at the commas that are not in quotes or parentheses."""
    parts: list[str] = []
    depth = 0
    quote = None
    start = 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"" and not (ch == "'" and text[max(0, i - 2):i].lower() == "af"):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    if parts == [""]:
        return []
    return parts


@dataclass(frozen=True)
class Operand:
    """One classified operand."""

    kind: str  # r8 r16 i r mem_hl mem_bc mem_de mem_sp mem_idx mem_abs port_c imm
    name: str = ""  # register name, or index register for mem_idx
    text: str = ""  # original text

    @property
    def regs(self) -> frozenset[str]:
        """Registers read to form the value or address of this operand."""
        if self.kind == "r8":
            return frozenset({self.name})
        if self.kind == "r16":
            return PAIRS[self.name]
        if self.kind == "mem_hl":
            return PAIRS["hl"]
        if self.kind == "mem_bc":
            return PAIRS["bc"]
        if self.kind == "mem_de":
            return PAIRS["de"]
        if self.kind == "mem_sp":
            return PAIRS["sp"]
        if self.kind == "mem_idx":
            return PAIRS[self.name]
        if self.kind == "port_c":
            return PAIRS["bc"]
        if self.kind in ("i", "r"):
            return frozenset({self.kind})
        return frozenset()


_IDX = re.compile(r"^\((ix|iy)([+-].*)?\)$")


def _fits(text: str, lo: int, hi: int) -> bool:
    """False only for a numeric literal outside ``lo..hi``."""
    v = parse_number(text)
    return v is None or lo <= v <= hi


def _disp_ok(low: str) -> bool:
    """An index displacement that is a literal must be a signed byte."""
    m = _IDX.match(low)
    if not m or not m.group(2):
        return True
    return _fits(m.group(2).replace("+-", "-").lstrip("+"), -128, 127)


def classify(text: str) -> Operand:
    """Classify one operand of an instruction."""
    t = text.strip()
    low = re.sub(r"\s+", "", t.lower())
    if low in REGS8:
        return Operand("r8", low, t)
    if low in PAIRS or low == "af'":
        return Operand("r16", "af" if low == "af'" else low, t)
    if low in ("i", "r"):
        return Operand(low, low, t)
    if low.startswith("(") and low.endswith(")") and _balanced_outer(low):
        inner = low[1:-1]
        if inner == "hl":
            return Operand("mem_hl", "hl", t)
        if inner == "bc":
            return Operand("mem_bc", "bc", t)
        if inner == "de":
            return Operand("mem_de", "de", t)
        if inner == "sp":
            return Operand("mem_sp", "sp", t)
        if inner == "c":
            return Operand("port_c", "c", t)
        m = _IDX.match(low)
        if m:
            if not _disp_ok(low):
                return Operand("bad", "", t)
            return Operand("mem_idx", m.group(1), t)
        if _names_register(inner):
            return Operand("bad", "", t)
        return Operand("mem_abs", "", t)
    if _names_register(low):
        return Operand("bad", "", t)
    return Operand("imm", "", t)


def _balanced_outer(s: str) -> bool:
    """True if the parentheses at both ends of ``s`` enclose all of it."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


_REGWORD = re.compile(r"(?<![\w?@$.])(a|b|c|d|e|h|l|af|bc|de|hl|sp|ix|iy|ixh|ixl|iyh|iyl)(?![\w?@$.'])")


def _names_register(expr: str) -> bool:
    """Does an expression mention a register (``ix+5`` outside parentheses)?"""
    return bool(_REGWORD.search(expr))


def _alu_src(op: Operand) -> tuple[frozenset[str], int] | None:
    """Reads and size of the source operand of an eight-bit ALU instruction."""
    if op.kind == "r8":
        return op.regs, 1
    if op.kind == "mem_hl":
        return op.regs, 1
    if op.kind == "mem_idx":
        return op.regs, 3
    if op.kind == "imm":
        if not _fits(op.text, -128, 0xFF):
            return None
        return frozenset(), 2
    return None


def effect(opcode: str, operand_text: str, radix: int | None = 10) -> Effect:
    """What ``opcode operand_text`` reads, writes and does to control flow.

    ``radix`` is the text's default radix, as for :func:`parse_number`; only
    the size of ``ds`` depends on it."""
    op = opcode.lower()
    ops = split_operands(operand_text)
    try:
        return _effect(op, ops, radix)
    except (KeyError, IndexError, ValueError):
        return UNKNOWN


def _nx(reads, writes, size, **kw) -> Effect:
    return Effect(frozenset(reads), frozenset(writes), "next", None, size, **kw)


def _effect(op: str, ops: list[str], radix: int | None = 10) -> Effect:  # noqa: C901 - one table
    n = len(ops)
    if op in TRANSPARENT:
        return Effect(frozenset(), frozenset(), "next", None, 0)
    if op in DATA:
        return Effect(ALL, frozenset(), "data", None, data_size(op, ops, radix))
    if op in BARRIERS:
        return UNKNOWN

    if op == "ld" and n == 2:
        d, s = classify(ops[0]), classify(ops[1])
        if s.kind == "imm":
            wide = d.kind == "r16"
            if not _fits(s.text, -32768 if wide else -128, 0xFFFF if wide else 0xFF):
                return UNKNOWN
        return _ld(d, s)

    if op == "push" and n == 1:
        r = classify(ops[0])
        if r.kind == "r16" and r.name != "sp":
            return _nx(r.regs | {"sp"}, {"sp"}, 2 if r.name in ("ix", "iy") else 1,
                       stack=1, pair=r.name)
        return UNKNOWN
    if op == "pop" and n == 1:
        r = classify(ops[0])
        if r.kind == "r16" and r.name != "sp":
            return _nx({"sp"}, r.regs | {"sp"}, 2 if r.name in ("ix", "iy") else 1,
                       stack=-1, pair=r.name)
        return UNKNOWN

    if op == "ex" and n == 2:
        a, b = (x.lower().replace(" ", "") for x in ops)
        if (a, b) == ("de", "hl"):
            return Effect(frozenset(), frozenset(), "next", None, 1, swap_de_hl=True)
        if a == "af" and b == "af'":
            return _nx(PAIRS["af"], PAIRS["af"], 1)
        if a == "(sp)" and b in ("hl", "ix", "iy"):
            return _nx(PAIRS[b] | {"sp"}, PAIRS[b], 1 if b == "hl" else 2)
        return UNKNOWN
    if op == "exx" and n == 0:
        regs = PAIRS["bc"] | PAIRS["de"] | PAIRS["hl"]
        return _nx(regs, regs, 1)

    if op in ("add", "adc", "sub", "sbc", "and", "or", "xor", "cp"):
        return _alu(op, ops)

    if op in ("inc", "dec") and n == 1:
        r = classify(ops[0])
        if r.kind == "r8":
            return _nx(r.regs, r.regs | FLAGS_NO_C, 1)
        if r.kind == "mem_hl":
            return _nx(r.regs, FLAGS_NO_C, 1)
        if r.kind == "mem_idx":
            return _nx(r.regs, FLAGS_NO_C, 3)
        if r.kind == "r16" and r.name != "af":
            return _nx(r.regs, r.regs, 2 if r.name in ("ix", "iy") else 1,
                       stack=None if r.name == "sp" else 0)
        return UNKNOWN

    if n == 0:
        simple = {
            "nop": ((), (), 1),
            "di": ((), (), 1),
            "ei": ((), (), 1),
            "daa": (("a", "fc", "fh", "fn"), ("a", "fs", "fz", "fh", "fp", "fc"), 1),
            "cpl": (("a",), ("a", "fh", "fn"), 1),
            "neg": (("a",), {"a"} | FLAGS, 2),
            "ccf": (("fc",), ("fc", "fh", "fn"), 1),
            "scf": ((), ("fc", "fh", "fn"), 1),
            "rlca": (("a",), ("a", "fh", "fn", "fc"), 1),
            "rrca": (("a",), ("a", "fh", "fn", "fc"), 1),
            "rla": (("a", "fc"), ("a", "fh", "fn", "fc"), 1),
            "rra": (("a", "fc"), ("a", "fh", "fn", "fc"), 1),
            "rld": (("a", "h", "l"), {"a"} | FLAGS_NO_C, 2),
            "rrd": (("a", "h", "l"), {"a"} | FLAGS_NO_C, 2),
            "ldi": (("b", "c", "d", "e", "h", "l"), ("b", "c", "d", "e", "h", "l", "fh", "fp", "fn"), 2),
            "ldd": (("b", "c", "d", "e", "h", "l"), ("b", "c", "d", "e", "h", "l", "fh", "fp", "fn"), 2),
            "ldir": (("b", "c", "d", "e", "h", "l"), ("b", "c", "d", "e", "h", "l", "fh", "fp", "fn"), 2),
            "lddr": (("b", "c", "d", "e", "h", "l"), ("b", "c", "d", "e", "h", "l", "fh", "fp", "fn"), 2),
            "cpi": (("a", "b", "c", "h", "l"), {"b", "c", "h", "l"} | FLAGS_NO_C, 2),
            "cpd": (("a", "b", "c", "h", "l"), {"b", "c", "h", "l"} | FLAGS_NO_C, 2),
            "cpir": (("a", "b", "c", "h", "l"), {"b", "c", "h", "l"} | FLAGS_NO_C, 2),
            "cpdr": (("a", "b", "c", "h", "l"), {"b", "c", "h", "l"} | FLAGS_NO_C, 2),
        }
        if op in simple:
            reads, writes, size = simple[op]
            return _nx(reads, writes, size)
        if op in ("ini", "ind", "inir", "indr", "outi", "outd", "otir", "otdr"):
            return _nx({"b", "c", "h", "l"}, {"b", "h", "l", "fz", "fn"}, 2)
        if op == "halt":
            return Effect(ALL, frozenset(), "stop", None, 1)
        if op == "ret":
            return Effect(frozenset(), frozenset(), "return", None, 1)
        if op in ("reti", "retn"):
            return Effect(ALL, frozenset(), "stop", None, 2)

    if op in ("rlc", "rrc", "rl", "rr", "sla", "sra", "srl", "sll", "sl1") and n == 1:
        r = classify(ops[0])
        carry_in = {"fc"} if op in ("rl", "rr") else set()
        if r.kind == "r8":
            return _nx(r.regs | carry_in, r.regs | FLAGS, 2)
        if r.kind == "mem_hl":
            return _nx(r.regs | carry_in, FLAGS, 2)
        if r.kind == "mem_idx":
            return _nx(r.regs | carry_in, FLAGS, 4)
        return UNKNOWN

    if op in ("bit", "set", "res") and n == 2:
        if not ops[0].strip().isdigit() or int(ops[0]) > 7:
            return UNKNOWN
        r = classify(ops[1])
        if r.kind == "r8":
            size = 2
        elif r.kind == "mem_hl":
            size = 2
        elif r.kind == "mem_idx":
            size = 4
        else:
            return UNKNOWN
        if op == "bit":
            # S and P/V are undefined after BIT; only Z, H and N are sure.
            return _nx(r.regs, {"fz", "fh", "fn"}, size)
        writes = r.regs if r.kind == "r8" else frozenset()
        return _nx(r.regs, writes, size)

    if op == "im" and n == 1:
        return _nx((), (), 2)

    if op == "jp":
        if n == 1:
            t = classify(ops[0])
            if t.kind in ("mem_hl", "mem_idx") and t.text.replace(" ", "").lower() in ("(hl)", "(ix)", "(iy)"):
                return Effect(ALL, frozenset(), "stop", None, 1 if t.kind == "mem_hl" else 2)
            if t.kind == "imm":
                return Effect(frozenset(), frozenset(), "jump", ops[0].strip(), 3)
            return UNKNOWN
        if n == 2 and ops[0].lower() in CONDITIONS and classify(ops[1]).kind == "imm":
            return Effect(CONDITIONS[ops[0].lower()], frozenset(), "branch", ops[1].strip(), 3,
                          cond=True)
        return UNKNOWN

    if op == "jr":
        if n == 1 and classify(ops[0]).kind == "imm":
            return Effect(frozenset(), frozenset(), "jump", ops[0].strip(), 2)
        if n == 2 and ops[0].lower() in JR_CONDITIONS and classify(ops[1]).kind == "imm":
            return Effect(CONDITIONS[ops[0].lower()], frozenset(), "branch", ops[1].strip(), 2,
                          cond=True)
        return UNKNOWN

    if op == "djnz" and n == 1 and classify(ops[0]).kind == "imm":
        return Effect(frozenset({"b"}), frozenset({"b"}), "branch", ops[0].strip(), 2, cond=True)

    if op == "call":
        if n == 1 and classify(ops[0]).kind == "imm":
            return Effect(frozenset(), frozenset(), "call", ops[0].strip(), 3)
        if n == 2 and ops[0].lower() in CONDITIONS and classify(ops[1]).kind == "imm":
            return Effect(CONDITIONS[ops[0].lower()], frozenset(), "call", ops[1].strip(), 3,
                          cond=True)
        return UNKNOWN

    if op == "ret" and n == 1 and ops[0].lower() in CONDITIONS:
        return Effect(CONDITIONS[ops[0].lower()], frozenset(), "return", None, 1, cond=True)

    if op == "rst" and n == 1:
        return Effect(frozenset(), frozenset(), "call", None, 1)

    if op == "in" and n == 2:
        d, s = classify(ops[0]), classify(ops[1])
        if d.kind == "r8" and d.name == "a" and s.kind == "mem_abs":
            return _nx({"a"}, {"a"}, 2)
        if d.kind == "r8" and d.name in REGS8 and s.kind == "port_c":
            return _nx(s.regs, d.regs | FLAGS_NO_C, 2)
        return UNKNOWN
    if op == "out" and n == 2:
        d, s = classify(ops[0]), classify(ops[1])
        if d.kind == "mem_abs" and s.kind == "r8" and s.name == "a":
            return _nx({"a"}, (), 2)
        if d.kind == "port_c" and s.kind == "r8" and s.name in REGS8:
            return _nx(d.regs | s.regs, (), 2)
        return UNKNOWN

    return UNKNOWN


def _ld(d: Operand, s: Operand) -> Effect:
    """``ld d,s``."""
    # Eight-bit register destination.
    if d.kind == "r8":
        if s.kind == "r8":
            return _nx(s.regs, d.regs, 1)
        if s.kind == "imm":
            return _nx((), d.regs, 2)
        if s.kind == "mem_hl":
            return _nx(s.regs, d.regs, 1)
        if s.kind == "mem_idx":
            return _nx(s.regs, d.regs, 3)
        if d.name == "a":
            if s.kind in ("mem_bc", "mem_de"):
                return _nx(s.regs, d.regs, 1)
            if s.kind == "mem_abs":
                return _nx((), d.regs, 3)
            if s.kind in ("i", "r"):
                return _nx(s.regs, d.regs | FLAGS_NO_C, 2)
        return UNKNOWN
    if d.kind in ("i", "r"):
        if s.kind == "r8" and s.name == "a":
            return _nx({"a"}, d.regs, 2)
        return UNKNOWN
    # Memory destination.
    if d.kind == "mem_hl":
        if s.kind == "r8":
            return _nx(d.regs | s.regs, (), 1)
        if s.kind == "imm":
            return _nx(d.regs, (), 2)
        return UNKNOWN
    if d.kind == "mem_idx":
        if s.kind == "r8":
            return _nx(d.regs | s.regs, (), 3)
        if s.kind == "imm":
            return _nx(d.regs, (), 4)
        return UNKNOWN
    if d.kind in ("mem_bc", "mem_de"):
        if s.kind == "r8" and s.name == "a":
            return _nx(d.regs | s.regs, (), 1)
        return UNKNOWN
    if d.kind == "mem_abs":
        if s.kind == "r8" and s.name == "a":
            return _nx({"a"}, (), 3)
        if s.kind == "r16" and s.name != "af":
            return _nx(s.regs, (), 3 if s.name == "hl" else 4)
        return UNKNOWN
    # Sixteen-bit register destination.
    if d.kind == "r16" and d.name != "af":
        stack = None if d.name == "sp" else 0
        if s.kind == "imm":
            return _nx((), d.regs, 4 if d.name in ("ix", "iy") else 3, stack=stack)
        if s.kind == "mem_abs":
            return _nx((), d.regs, 3 if d.name == "hl" else 4, stack=stack)
        if d.name == "sp" and s.kind == "r16" and s.name in ("hl", "ix", "iy"):
            return _nx(s.regs, d.regs, 1 if s.name == "hl" else 2, stack=None)
        return UNKNOWN
    return UNKNOWN


def _alu(op: str, ops: list[str]) -> Effect:
    """Eight-bit ALU instructions, and add/adc/sbc on HL, IX and IY."""
    n = len(ops)
    if n == 2:
        d = classify(ops[0])
        s = classify(ops[1])
        if d.kind == "r16" and d.name in ("hl", "ix", "iy") and s.kind == "r16":
            if op == "add":
                if s.name not in ("bc", "de", "sp", d.name):
                    return UNKNOWN
                return _nx(d.regs | s.regs, d.regs | {"fh", "fn", "fc"}, 1 if d.name == "hl" else 2)
            if op in ("adc", "sbc") and d.name == "hl" and s.name in ("bc", "de", "hl", "sp"):
                return _nx(d.regs | s.regs | {"fc"}, d.regs | FLAGS, 2)
            return UNKNOWN
        if not (d.kind == "r8" and d.name == "a"):
            return UNKNOWN
        src = s
    elif n == 1:
        src = classify(ops[0])
    else:
        return UNKNOWN
    got = _alu_src(src)
    if got is None:
        return UNKNOWN
    reads, size = got
    reads = set(reads) | {"a"}
    if src.kind == "r8" and src.name == "a" and op in ("xor", "sub", "cp", "sbc"):
        reads = set()  # the result does not depend on A
    if op in ("adc", "sbc"):
        reads.add("fc")
    writes = set(FLAGS)
    if op != "cp":
        writes.add("a")
    return _nx(reads, writes, size)


def data_size(op: str, ops: list[str], radix: int | None = 10) -> int | None:
    """Bytes emitted by a data directive, or None if that is not known here."""
    if op in ("db", "defb", "defm", "byte", "dc"):
        total = 0
        for item in ops:
            it = item.strip()
            if len(it) >= 2 and it[0] == it[-1] and it[0] in "'\"":
                body = it[1:-1].replace(it[0] * 2, it[0])
                if op == "dc":
                    return None
                total += len(body)
            else:
                total += 1
        return total
    if op in ("dw", "defw", "word"):
        return 2 * len(ops)
    if op in ("ds", "defs"):
        if not ops:
            return None
        v = parse_number(ops[0], radix)
        return v if v is not None and v >= 0 else None
    return None


def parse_number(text: str, radix: int | None = 10) -> int | None:
    """The value of a numeric literal (decimal, ``0FFH``, ``0x10``, ``101B``,
    ``17O``/``17Q``, optionally negated), or None.

    ``radix`` is the default radix of the text.  None stands for one that is
    not known, or not ten (after ``.radix 16``, ``300`` is 300H and ``101B``
    and ``12D`` are hexadecimal too): then only a number whose value is the
    same under every radix has one - a single digit, or one with the suffix
    H, O or Q."""
    s = text.strip().upper().replace(" ", "")
    if not s:
        return None
    neg = False
    if s[0] in "+-":
        neg = s[0] == "-"
        s = s[1:]
    if not s or not s[0].isdigit():
        return None
    if radix != 10 and not (len(s) == 1 or s.endswith(("H", "O", "Q"))):
        return None
    try:
        if s.startswith("0X"):
            v = int(s[2:], 16)
        elif s.endswith("H"):
            v = int(s[:-1], 16)
        elif s.endswith("B") and all(c in "01" for c in s[:-1]) and len(s) > 1:
            v = int(s[:-1], 2)
        elif s.endswith(("O", "Q")):
            v = int(s[:-1], 8)
        elif s.endswith("D"):
            v = int(s[:-1], 10)
        else:
            v = int(s, 10)
    except ValueError:
        return None
    return -v if neg else v


def hex_byte(v: int) -> str:
    """``v``'s low byte as a number that reads the same under any radix."""
    return f"0{v & 0xFF:02X}h"
