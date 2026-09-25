"""
Peephole Optimizer for Z80.

Performs pattern-based optimizations on generated Z80 assembly code.
This runs after code generation to clean up inefficient sequences.

This module expects pure Z80 mnemonics as input (ld, jp, jr, etc.)
and produces optimized Z80 assembly as output. All output uses
lowercase mnemonics and register names.

A rewrite that changes what a register or flag holds afterwards is made only
where nothing reads the old value: the optimizer follows every path from the
end of the rewritten code - on, into both arms of a branch, round loops - and
the value must be overwritten before it is read on each.  A path that leaves
the code the optimizer can see (a call, a return, ``jp (hl)``, a jump to a
label defined elsewhere, data, the end of the text) counts as reading
everything, since the caller, the callee or the other module may.

For compilers that generate 8080 mnemonics, use upeep80 instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from .z80 import (
    PAIRS as _PAIRS,
    BARRIERS,
    DATA,
    FLAGS,
    FLAGS_NO_C,
    JR_CONDITIONS,
    TRANSPARENT,
    UNKNOWN,
    Effect,
    classify,
    effect,
    parse_number,
    split_operands,
    strip_comment,
)

Instr = tuple[str, str]

# What a call to the runtime multiply or subtract may change besides HL.
_NOT_HL = frozenset("abcde") | FLAGS
_MUL16 = ("??mul16", "@mul16", "__mul16")
_SUBDE = ("??subde", "@subde")

_LABEL = re.compile(r"^([A-Za-z_?@$.][\w?@$.]*)(::?)?(.*)$")
# `NAME::' is M80's way to define a label and make it PUBLIC in one.
_EXPORTED = re.compile(r"^([A-Za-z_?@$.][\w?@$.]*)::")
_IDENT = re.compile(r"[A-Za-z_?@$.][\w?@$.]*")
_SWAP = {"d": "h", "e": "l", "h": "d", "l": "e"}
# Directives that give a name a value.  Only an `equ' gives it one value
# throughout; the others may be redefined further down.
_EQUATES = ("equ", "defl", "set", "=")


@lru_cache(maxsize=1 << 16)
def _split(line: str) -> tuple[str | None, str | None, str]:
    """``(label, opcode, operands)`` of one line; opcode is lowercase and the
    operands have no blanks around their commas.  ``NAME equ VALUE`` in
    column 1 is a label with the opcode ``equ`` (or ``defl``, ``set``, ``=``)."""
    text = strip_comment(line)
    if not text.strip():
        return None, None, ""
    label = None
    body = text
    if not text[0].isspace():
        m = _LABEL.match(text)
        if m and m.group(2):
            label, body = m.group(1), m.group(3)
        elif m:
            rest = m.group(3).split(None, 1)
            if rest and rest[0].lower() in _EQUATES:
                return m.group(1), rest[0].lower(), rest[1].strip() if len(rest) > 1 else ""
    parts = body.split(None, 1)
    if not parts:
        return label, None, ""
    operands = ",".join(split_operands(parts[1])) if len(parts) > 1 else ""
    return label, parts[0].lower(), operands


def _exported(line: str) -> str | None:
    """The label ``line`` defines with ``::``, which exports it, or None."""
    m = _EXPORTED.match(line)
    return m.group(1) if m else None


def _same_operand(a: str, b: str) -> bool:
    return a.replace(" ", "").lower() == b.replace(" ", "").lower()


# How many calls deep liveness follows into routines of the module, and how
# many instructions one question may look at before it gives up and answers
# "live".
_MAX_FRAMES = 4
_BUDGET = 50000

# The halves of a register pair as it is pushed: (high byte, low byte).
_HALVES = {"af": ("a", None), "bc": ("b", "c"), "de": ("d", "e"), "hl": ("h", "l"),
           "ix": ("ixh", "ixl"), "iy": ("iyh", "iyl")}


def _slots(need: frozenset) -> bool:
    return any(type(x) is tuple for x in need)


class _Routines:
    """Where each line's ``ret`` goes, and how deep the stack is there.

    A routine is a label that a ``call`` names and nothing else does - no
    ``ld hl,L``, ``dw L``, ``public L``, ``L::`` or ``X equ L``, and no jump
    spelled differently from the label.  Its code is what can be reached
    from the label without going into a call.  A ``ret`` in code that only
    routines reach returns to the line after one of their calls.  Code that
    can be reached any other way - from the first line, from a label
    something names or the module exports, from after data, a directive,
    ``jp (hl)`` or an instruction not recognised - may have been entered
    from anywhere, and so may its ``ret`` go anywhere.

    ``height`` is the number of pushes outstanding since the routine was
    entered, where every way to a line agrees on it, else None.

    Code that takes its return address off the stack - a ``pop`` or ``ex
    (sp),rr`` where nothing of its own is pushed - may put another in its
    place, or none (``pop hl / push de / ret``, ``ex (sp),hl / ret``).  Its
    ``ret`` goes anywhere: ``wild`` is set for all the code that can be
    reached from where it was entered.  (Where the height is not known, a
    ``ret`` goes anywhere anyway.)  Such a routine, and one with a ``ret``
    where the height is not known, may come back to its call with the stack
    as it pleases: they are ``irregular``, and the height after a call of
    one is not known.
    """

    def __init__(self, code: "_Code"):
        effects = code.effects
        n = len(effects)
        named: set[str] = set()
        for idx, line in enumerate(code.lines):
            _, op, operands = _split(line)
            if op is None or not operands:
                continue
            eff = effects[idx]
            direct = None
            if eff is not None and eff.flow in ("jump", "branch", "call") and \
                    code.target(eff.target) is not None:
                direct = eff.target
            for name in _IDENT.findall(operands):
                if name != direct:
                    named.add(name.lower())
        self.calls: dict[int, list[int]] = {}
        for idx, eff in enumerate(effects):
            if eff is not None and eff.flow == "call":
                t = code.target(eff.target)
                if t is not None:
                    self.calls.setdefault(t, []).append(idx + 1)
        open_roots = {0}
        for name, idx in code.label_lines:
            if name.lower() in named or name.lower() in code.exported or \
                    code.labels.get(name) is None:
                open_roots.add(idx)
        for idx, eff in enumerate(effects):
            if eff is not None and eff.flow in ("stop", "data"):
                open_roots.add(idx + 1)

        def successors(i: int) -> list[int]:
            eff = effects[i]
            if eff is None or eff.flow == "next" or eff.flow == "call":
                return [i + 1]
            if eff.flow == "jump":
                t = code.target(eff.target)
                return [] if t is None else [t]
            if eff.flow == "branch":
                t = code.target(eff.target)
                return [i + 1] if t is None else [i + 1, t]
            if eff.flow == "return" and eff.cond:
                return [i + 1]
            return []

        self.open = [False] * (n + 1)
        self.entries: list[set[int]] = [set() for _ in range(n + 1)]
        roots = sorted(open_roots | set(self.calls))
        reach: dict[int, set[int]] = {}
        for root in roots:
            is_open = root in open_roots
            stack = [root]
            reached: set[int] = set()
            while stack:
                i = stack.pop()
                if i in reached or i >= n:
                    continue
                reached.add(i)
                if is_open:
                    self.open[i] = True
                else:
                    self.entries[i].add(root)
                stack.extend(successors(i))
            reach[root] = reached

        # Routines that move their return address or return at a height not
        # known, and the heights, which are unknown after a call of one:
        # each can make more of the other.
        unbalanced: set[int] = set()
        callees = set(self.calls)
        while True:
            height = self._heights(code, roots, successors, unbalanced)
            wild = {root for root in roots
                    if any(self._moves_return(code, i, height[i]) for i in reach[root])}
            irregular = wild | {root for root in roots
                                if any(height[i] is None and effects[i] is not None and
                                       effects[i].flow == "return" for i in reach[root])}
            more = (irregular & callees) - unbalanced
            if not more:
                break
            unbalanced |= more
        self.height: list[int | None] = height
        self.irregular = irregular
        self.wild = [False] * (n + 1)
        for root in wild:
            for i in reach[root]:
                self.wild[i] = True

    @staticmethod
    def _heights(code: "_Code", roots: list[int], successors: Callable[[int], list[int]],
                 unbalanced: set[int]) -> list[int | None]:
        """Stack heights: 0 where a routine (or anything else) is entered."""
        effects = code.effects
        n = len(effects)
        unset = object()
        height: list = [unset] * (n + 1)
        work = []
        for root in roots:
            if root < n:
                height[root] = 0
                work.append(root)
        while work:
            i = work.pop()
            h = height[i]
            eff = effects[i]
            if h is None or eff is None:
                after = h
            elif eff.stack is None:
                after = None
            elif eff.flow == "call" and code.target(eff.target) in unbalanced:
                after = None
            else:
                after = h + eff.stack
            for j in successors(i):
                if j >= n:
                    continue
                old = height[j]
                if old is unset:
                    height[j] = after
                elif old is None or old == after:
                    continue
                else:
                    height[j] = None  # two ways in disagree
                work.append(j)
        return [None if h is unset else h for h in height]

    @staticmethod
    def _moves_return(code: "_Code", i: int, h: int | None) -> bool:
        """Does line ``i``, at stack height ``h``, take the return address
        (or what is under it) off the stack?"""
        eff = code.effects[i]
        if h is None or eff is None or eff.flow != "next" or "sp" not in eff.reads:
            return False
        return h <= 0 and (eff.stack == -1 or (eff.stack == 0 and _split(code.lines[i])[1] == "ex"))

    def continuations(self, i: int) -> list[int] | None:
        """The lines a ``ret`` at line ``i`` may return to, or None if any."""
        if self.open[i] or self.wild[i] or not self.entries[i]:
            return None
        out: list[int] = []
        for root in self.entries[i]:
            out.extend(self.calls[root])
        return out


class _Code:
    """One version of the program: what each line does, and where labels are."""

    def __init__(self, lines: list[str]):
        self.lines = lines
        self.effects: list[Effect | None] = []
        self.labels: dict[str, int | None] = {}
        self.label_lines: list[tuple[str, int]] = []
        self.equ: dict[str, int] = {}
        # Labels defined with `::', lowercase.
        self.exported: set[str] = set()
        seen_equ: set[str] = set()
        for idx, line in enumerate(lines):
            name = _exported(line)
            if name:
                self.exported.add(name.lower())
            label, op, operands = _split(line)
            if op in _EQUATES and label and (op != "set" or "," not in operands):
                v = parse_number(operands)
                if op == "equ" and v is not None and label not in seen_equ:
                    self.equ[label] = v
                else:
                    self.equ.pop(label, None)
                seen_equ.add(label)
                self.effects.append(None)
                continue
            if label:
                # A label defined twice is nowhere in particular.
                self.labels[label] = None if label in self.labels else idx
                self.label_lines.append((label, idx))
            self.effects.append(None if op is None else effect(op, operands))
        self._layout: tuple[list[int], list[int], list[int]] | None = None
        self._routines: _Routines | None = None

    def target(self, name: str | None) -> int | None:
        if name is None:
            return None
        return self.labels.get(name)

    @property
    def routines(self) -> _Routines:
        if self._routines is None:
            self._routines = _Routines(self)
        return self._routines

    def live(self, starts: list[int], resources: frozenset[str] | set[str]) -> bool:
        """May any of ``resources``, as they are at the lines ``starts``, be
        read on some path from there before it is written?

        Every path is followed: on, into both arms of a branch, round loops,
        into a routine of the module that is called and back after the call,
        and from a ``ret`` to the line after each call of the routine.  A
        value pushed is followed through its slot on the stack to the pop
        that takes it off.  What cannot be followed - a call or jump out of
        the module, ``jp (hl)``, a ``ret`` from code entered who knows how,
        data, the end of the text, a stack that is not balanced - reads
        everything."""
        need0 = frozenset(resources)
        if not need0:
            return False
        routines = self.routines
        height = routines.height
        n = len(self.effects)
        work: list[tuple[int | None, frozenset, tuple[int, ...]]] = [(s, need0, ()) for s in starts]
        seen: dict[tuple[int, tuple[int, ...]], list[frozenset]] = {}
        budget = _BUDGET
        while work:
            i, need, frames = work.pop()
            while True:
                if i is None or i >= n:
                    return True
                eff = self.effects[i]
                if eff is None:
                    i += 1
                    continue
                budget -= 1
                if budget < 0:
                    return True
                key = (i, frames)
                prior = seen.setdefault(key, [])
                if any(need <= p for p in prior):
                    break
                prior.append(need)
                depth = len(frames)
                h = height[i]

                if eff.stack == 1 and eff.pair:
                    # push: the value is still in the register, and now in a
                    # stack slot as well.
                    hit = need & _PAIRS[eff.pair]
                    if hit:
                        if h is None:
                            return True
                        hi, _ = _HALVES[eff.pair]
                        need = frozenset(x for x in need
                                         if not (type(x) is tuple and x[:2] == (depth, h + 1)))
                        need = need | {(depth, h + 1, "hi" if x == hi else "lo",
                                        x if eff.pair == "af" and x != "a" else None)
                                       for x in hit}
                    i += 1
                    continue
                if eff.stack == -1 and eff.pair:
                    # pop: what was in the slot is in the register now.
                    need = need - _PAIRS[eff.pair]
                    if _slots(need):
                        if h is None or h <= 0:
                            return True
                        hi, lo = _HALVES[eff.pair]
                        mine = {x for x in need if type(x) is tuple and x[:2] == (depth, h)}
                        need = need - mine
                        for _, _, half, piece in mine:
                            if half == "hi":
                                need = need | {hi}
                            elif eff.pair == "af":
                                need = need | ({piece} if piece else FLAGS)
                            else:
                                need = need | {lo}
                    if not need:
                        break
                    i += 1
                    continue

                if eff.reads & need:
                    return True
                if eff.flow == "call":
                    t = self.target(eff.target)
                    if t is None or depth >= _MAX_FRAMES:
                        return True
                    if eff.cond:
                        work.append((i + 1, need, frames))
                    frames = frames + (i + 1,)
                    i = t
                    continue
                if eff.flow == "return":
                    if eff.cond:
                        work.append((i + 1, need, frames))
                    if h != 0 or routines.wild[i]:
                        return True
                    # This routine's slots are below SP now.
                    need = frozenset(x for x in need if not (type(x) is tuple and x[0] == depth))
                    if not need:
                        break
                    if frames:
                        i = frames[-1]
                        frames = frames[:-1]
                        continue
                    conts = routines.continuations(i)
                    if conts is None:
                        return True
                    work.extend((c, need, ()) for c in conts)
                    break
                if (eff.stack is None or "sp" in eff.reads) and _slots(need):
                    return True
                if eff.swap_de_hl:
                    need = frozenset(_SWAP.get(x, x) for x in need)
                need = need - eff.writes
                if not need:
                    break
                if eff.flow == "next":
                    i += 1
                elif eff.flow == "jump":
                    i = self.target(eff.target)
                elif eff.flow == "branch":
                    t = self.target(eff.target)
                    if t is None:
                        return True
                    work.append((t, need, frames))
                    i += 1
                else:
                    return True
        return False

    def moves_return(self, name: str) -> bool:
        """Is ``name`` a routine of this text that may move its return
        address (or what the stack holds under it), or return at a stack
        height not known?"""
        t = self.target(name.strip())
        return t is not None and (t in self.routines.irregular or self.routines.wild[t])

    def layout(self) -> tuple[list[int], list[int], list[int]]:
        """Address, size and segment of every line.  A line whose size is not
        known here (a macro, ``ds`` of a symbol, ``org``) ends a segment, and
        distances are only measured within one."""
        if self._layout is None:
            addr, size, seg = [], [], []
            a = s = 0
            for eff in self.effects:
                n = 0 if eff is None else eff.size
                addr.append(a)
                seg.append(s)
                size.append(n or 0)
                if n is None:
                    s += 1
                    a = 0
                else:
                    a += n
            self._layout = (addr, size, seg)
        return self._layout

    def reaches(self, first: int, last: int, target: int, new_size: int = 2) -> bool:
        """Can lines ``first``..``last``, replaced by one relative jump of
        ``new_size`` bytes, reach line ``target``?  Measured on this version,
        which only shrinks from here: every rewrite after this point makes
        code shorter or leaves it the same length."""
        addr, size, seg = self.layout()
        if not (seg[first] == seg[last] == seg[target]):
            return False
        if target > last:
            return addr[target] - (addr[last] + size[last]) <= 127
        if target <= first:
            return addr[target] - (addr[first] + new_size) >= -128
        return False


@dataclass
class PeepholePattern:
    """A peephole optimization pattern."""

    name: str
    # Pattern: list of (opcode, operands) tuples, or regex strings
    pattern: list[tuple[str, str | None]]
    # Replacement: list of (opcode, operands) tuples, or None to delete
    replacement: list[tuple[str, str]] | None
    # Optional condition function
    condition: Callable[[list[tuple[str, str]]], bool] | None = None
    # Registers and flags (named as in upeepz80.z80) whose contents the
    # rewrite changes.  The pattern is applied only where each of them is
    # dead: overwritten before it is read on every path, from instruction
    # ``dead_from`` of the match on (by default, from the end of the match).
    # A callable is given the matched instructions.
    clobbers: frozenset[str] | Callable[[list[tuple[str, str]]], frozenset[str]] = frozenset()
    dead_from: int | None = None


def _duplicate_ld_ok(ops: list[Instr]) -> bool:
    """``ld x,y`` twice is ``ld x,y`` once unless the first changes ``y``:
    ``ld l,(hl)`` reads a different byte the second time."""
    if ops[0][1].lower() != ops[1][1].lower():
        return False
    eff = effect("ld", ops[0][1])
    return eff is not UNKNOWN and not (eff.reads & eff.writes) and "r" not in eff.reads


class PeepholeOptimizer:
    """
    Peephole optimizer for Z80 assembly.

    Applies pattern-based transformations to optimize Z80 code.
    Patterns are applied repeatedly until no more changes are made.

    This optimizer expects pure Z80 mnemonics (ld, jp, jr, etc.)
    as input and produces lowercase Z80 assembly output.
    """

    def __init__(self) -> None:
        self.patterns = self._init_patterns()
        self.stats: dict[str, int] = {}

    def _init_patterns(self) -> list[PeepholePattern]:
        """Initialize Z80 peephole optimization patterns."""
        hl = frozenset("hl")
        flags_no_c = FLAGS_NO_C
        return [
            # Push/Pop elimination: push rr; pop rr -> (nothing)
            PeepholePattern(
                name="push_pop_same",
                pattern=[("push", None), ("pop", None)],
                replacement=[],
                condition=lambda ops: ops[0][1].lower() == ops[1][1].lower(),
            ),
            # Redundant ld: ld a,r; ld r,a -> ld a,r
            PeepholePattern(
                name="redundant_ld",
                pattern=[("ld", "a,*"), ("ld", "*,a")],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1].split(",")[1].lower() == ops[1][1].split(",")[0].lower(),
            ),
            # Zero A: ld a,0 -> xor a (smaller, faster), which sets every flag
            PeepholePattern(
                name="zero_a_ld",
                pattern=[("ld", "a,0")],
                replacement=[("xor", "a")],
                clobbers=FLAGS,
            ),
            # Compare to zero: cp 0 -> or a.  S, Z, H and C come out the same;
            # P/V (overflow, not parity) and N do not.
            PeepholePattern(
                name="cp_zero",
                pattern=[("cp", "0")],
                replacement=[("or", "a")],
                clobbers=frozenset({"fp", "fn"}),
            ),
            # Redundant duplicate ld: ld x,y; ld x,y -> ld x,y
            PeepholePattern(
                name="duplicate_ld",
                pattern=[("ld", None), ("ld", None)],
                replacement=None,  # Keep first only
                condition=_duplicate_ld_ok,
            ),
            # ld r,r -> (nothing)
            PeepholePattern(
                name="ld_r_r",
                pattern=[("ld", None)],
                replacement=[],
                condition=lambda ops: len(ops[0][1].split(",")) == 2 and
                                      ops[0][1].split(",")[0].strip().lower() ==
                                      ops[0][1].split(",")[1].strip().lower() and
                                      ops[0][1].split(",")[0].strip().lower() in
                                      ("a", "b", "c", "d", "e", "h", "l"),
            ),
            # inc a; dec a -> (nothing); the flags are the dec's, not the old ones
            PeepholePattern(
                name="inc_dec_a",
                pattern=[("inc", "a"), ("dec", "a")],
                replacement=[],
                clobbers=flags_no_c,
            ),
            # dec a; inc a -> (nothing)
            PeepholePattern(
                name="dec_inc_a",
                pattern=[("dec", "a"), ("inc", "a")],
                replacement=[],
                clobbers=flags_no_c,
            ),
            # inc hl; dec hl -> (nothing)
            PeepholePattern(
                name="inc_dec_hl",
                pattern=[("inc", "hl"), ("dec", "hl")],
                replacement=[],
            ),
            # dec hl; inc hl -> (nothing)
            PeepholePattern(
                name="dec_inc_hl",
                pattern=[("dec", "hl"), ("inc", "hl")],
                replacement=[],
            ),
            # inc de; dec de -> (nothing)
            PeepholePattern(
                name="inc_dec_de",
                pattern=[("inc", "de"), ("dec", "de")],
                replacement=[],
            ),
            # dec de; inc de -> (nothing)
            PeepholePattern(
                name="dec_inc_de",
                pattern=[("dec", "de"), ("inc", "de")],
                replacement=[],
            ),
            # inc bc; dec bc -> (nothing)
            PeepholePattern(
                name="inc_dec_bc",
                pattern=[("inc", "bc"), ("dec", "bc")],
                replacement=[],
            ),
            # dec bc; inc bc -> (nothing)
            PeepholePattern(
                name="dec_inc_bc",
                pattern=[("dec", "bc"), ("inc", "bc")],
                replacement=[],
            ),
            # or a; or a -> or a
            PeepholePattern(
                name="double_or_a",
                pattern=[("or", "a"), ("or", "a")],
                replacement=[("or", "a")],
            ),
            # and a; and a -> and a
            PeepholePattern(
                name="double_and_a",
                pattern=[("and", "a"), ("and", "a")],
                replacement=[("and", "a")],
            ),
            # xor a; xor a -> xor a (still zero)
            PeepholePattern(
                name="double_xor_a",
                pattern=[("xor", "a"), ("xor", "a")],
                replacement=[("xor", "a")],
            ),
            # ex de,hl; ex de,hl -> (nothing)
            PeepholePattern(
                name="double_ex",
                pattern=[("ex", "de,hl"), ("ex", "de,hl")],
                replacement=[],
            ),
            # ex (sp),hl; ex (sp),hl -> (nothing)
            PeepholePattern(
                name="double_ex_sp",
                pattern=[("ex", "(sp),hl"), ("ex", "(sp),hl")],
                replacement=[],
            ),
            # ccf; ccf -> (nothing) - C is back, but H and N are the ccf's
            PeepholePattern(
                name="double_ccf",
                pattern=[("ccf", ""), ("ccf", "")],
                replacement=[],
                clobbers=frozenset({"fh", "fn"}),
            ),
            # cpl; cpl -> (nothing) - A is back, but cpl sets H and N
            PeepholePattern(
                name="double_cpl",
                pattern=[("cpl", ""), ("cpl", "")],
                replacement=[],
                clobbers=frozenset({"fh", "fn"}),
            ),
            # push hl; pop de -> ld d,h; ld e,l (faster: 21 cycles -> 8 cycles)
            PeepholePattern(
                name="push_pop_copy_hl_de",
                pattern=[("push", "hl"), ("pop", "de")],
                replacement=[("ld", "d,h"), ("ld", "e,l")],
            ),
            # push de; pop hl -> ld h,d; ld l,e
            PeepholePattern(
                name="push_pop_copy_de_hl",
                pattern=[("push", "de"), ("pop", "hl")],
                replacement=[("ld", "h,d"), ("ld", "l,e")],
            ),
            # push bc; pop de -> ld d,b; ld e,c
            PeepholePattern(
                name="push_pop_copy_bc_de",
                pattern=[("push", "bc"), ("pop", "de")],
                replacement=[("ld", "d,b"), ("ld", "e,c")],
            ),
            # push bc; pop hl -> ld h,b; ld l,c
            PeepholePattern(
                name="push_pop_copy_bc_hl",
                pattern=[("push", "bc"), ("pop", "hl")],
                replacement=[("ld", "h,b"), ("ld", "l,c")],
            ),
            # push hl; pop bc -> ld b,h; ld c,l
            PeepholePattern(
                name="push_pop_copy_hl_bc",
                pattern=[("push", "hl"), ("pop", "bc")],
                replacement=[("ld", "b,h"), ("ld", "c,l")],
            ),
            # push de; pop bc -> ld b,d; ld c,e
            PeepholePattern(
                name="push_pop_copy_de_bc",
                pattern=[("push", "de"), ("pop", "bc")],
                replacement=[("ld", "b,d"), ("ld", "c,e")],
            ),
            # ccf; scf -> scf (set carry directly)
            PeepholePattern(
                name="ccf_scf",
                pattern=[("ccf", None), ("scf", None)],
                replacement=[("scf", "")],
            ),
            # call x; ret -> jp x (tail call optimization).  Not `call cc,x':
            # when the call is not taken, the ret still has to happen.
            PeepholePattern(
                name="tail_call",
                pattern=[("call", None), ("ret", "")],
                replacement=None,  # Replaced specially
                condition=lambda ops: "," not in ops[0][1],
            ),
            # ret; ret -> ret (unreachable code)
            PeepholePattern(
                name="double_ret",
                pattern=[("ret", ""), ("ret", "")],
                replacement=[("ret", "")],
            ),
            # ld a,(hl); ld e,a -> ld e,(hl), where A is not read afterwards
            PeepholePattern(
                name="ld_a_hl_ld_ea",
                pattern=[("ld", "a,(hl)"), ("ld", "e,a")],
                replacement=[("ld", "e,(hl)")],
                clobbers=frozenset("a"),
            ),
            # ld a,(hl); ld d,a -> ld d,(hl)
            PeepholePattern(
                name="ld_a_hl_ld_da",
                pattern=[("ld", "a,(hl)"), ("ld", "d,a")],
                replacement=[("ld", "d,(hl)")],
                clobbers=frozenset("a"),
            ),
            # ld a,(hl); ld c,a -> ld c,(hl)
            PeepholePattern(
                name="ld_a_hl_ld_ca",
                pattern=[("ld", "a,(hl)"), ("ld", "c,a")],
                replacement=[("ld", "c,(hl)")],
                clobbers=frozenset("a"),
            ),
            # ld a,(hl); ld b,a -> ld b,(hl)
            PeepholePattern(
                name="ld_a_hl_ld_ba",
                pattern=[("ld", "a,(hl)"), ("ld", "b,a")],
                replacement=[("ld", "b,(hl)")],
                clobbers=frozenset("a"),
            ),
            # ld b,a; ld a,b -> ld b,a
            PeepholePattern(
                name="ld_ba_ab",
                pattern=[("ld", "b,a"), ("ld", "a,b")],
                replacement=[("ld", "b,a")],
            ),
            # ld c,a; ld a,c -> ld c,a
            PeepholePattern(
                name="ld_ca_ac",
                pattern=[("ld", "c,a"), ("ld", "a,c")],
                replacement=[("ld", "c,a")],
            ),
            # ld d,a; ld a,d -> ld d,a
            PeepholePattern(
                name="ld_da_ad",
                pattern=[("ld", "d,a"), ("ld", "a,d")],
                replacement=[("ld", "d,a")],
            ),
            # ld e,a; ld a,e -> ld e,a
            PeepholePattern(
                name="ld_ea_ae",
                pattern=[("ld", "e,a"), ("ld", "a,e")],
                replacement=[("ld", "e,a")],
            ),
            # ld h,a; ld a,h -> ld h,a
            PeepholePattern(
                name="ld_ha_ah",
                pattern=[("ld", "h,a"), ("ld", "a,h")],
                replacement=[("ld", "h,a")],
            ),
            # ld l,a; ld a,l -> ld l,a
            PeepholePattern(
                name="ld_la_al",
                pattern=[("ld", "l,a"), ("ld", "a,l")],
                replacement=[("ld", "l,a")],
            ),
            # ld (addr),hl; ld hl,(addr) -> ld (addr),hl (same address)
            PeepholePattern(
                name="ld_store_load_same",
                pattern=[("ld", None), ("ld", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: (ops[0][1].startswith("(") and
                                       ops[0][1].lower().endswith("),hl") and
                                       ops[1][1].lower() == f"hl,{ops[0][1][:-3].lower()}"),
            ),
            # ld (addr),a; ld a,(addr) -> ld (addr),a (same address)
            PeepholePattern(
                name="sta_lda_same",
                pattern=[("ld", None), ("ld", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: (ops[0][1].startswith("(") and
                                       ops[0][1].lower().endswith("),a") and
                                       ops[1][1].lower() == f"a,{ops[0][1][:-2].lower()}"),
            ),
            # and 0ffh -> or a (same effect, smaller) - but and sets H, or clears it
            PeepholePattern(
                name="and_ff",
                pattern=[("and", "0ffh")],
                replacement=[("or", "a")],
                clobbers=frozenset({"fh"}),
            ),
            # or 0 -> or a (same effect)
            PeepholePattern(
                name="or_0",
                pattern=[("or", "0")],
                replacement=[("or", "a")],
            ),
            # xor 0 -> or a (same effect, sets flags)
            PeepholePattern(
                name="xor_0",
                pattern=[("xor", "0")],
                replacement=[("or", "a")],
            ),
            # push hl; ex de,hl; pop hl -> ld d,h; ld e,l
            # The ex swaps hl<->de, then pop restores hl, so de = original hl
            PeepholePattern(
                name="push_ex_pop",
                pattern=[("push", "hl"), ("ex", "de,hl"), ("pop", "hl")],
                replacement=[("ld", "d,h"), ("ld", "e,l")],
            ),
            # ld h,0; ld d,h; ld e,l -> ld d,0; ld e,l, where H is not read
            # afterwards (the ld h,0 is gone)
            PeepholePattern(
                name="ld_h0_dh_el",
                pattern=[("ld", "h,0"), ("ld", "d,h"), ("ld", "e,l")],
                replacement=[("ld", "d,0"), ("ld", "e,l")],
                clobbers=frozenset("h"),
            ),
            # Wasteful byte extension before byte op: ld l,a; ld h,0; sub x -> sub x
            # (Also for cp.)  HL must be dead from the sub on - `sub l' reads it.
            PeepholePattern(
                name="useless_extend_before_sub",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("sub", None)],
                replacement=None,  # Keep last only
                clobbers=hl,
                dead_from=2,
            ),
            PeepholePattern(
                name="useless_extend_before_cp",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("cp", None)],
                replacement=None,  # Keep last only
                clobbers=hl,
                dead_from=2,
            ),
            # Redundant byte extension: ld l,a; ld h,0; ld l,a; ld h,0 -> ld l,a; ld h,0
            PeepholePattern(
                name="double_byte_extend",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("ld", "l,a"), ("ld", "h,0")],
                replacement=[("ld", "l,a"), ("ld", "h,0")],
            ),
            # Redundant load after push: ld l,a; ld h,0; push hl; ld l,a -> ld l,a; ld h,0; push hl
            PeepholePattern(
                name="redundant_ld_l_after_push",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("push", "hl"), ("ld", "l,a")],
                replacement=[("ld", "l,a"), ("ld", "h,0"), ("push", "hl")],
            ),
            # ld hl,0ffffh; ld a,l; or h -> ld a,0ffh; or a where HL is dead:
            # A is 0FFH and the flags are those of 0FFH either way.
            PeepholePattern(
                name="test_true_const",
                pattern=[("ld", "hl,0ffffh"), ("ld", "a,l"), ("or", "h")],
                replacement=[("ld", "a,0ffh"), ("or", "a")],
                clobbers=hl,
            ),
            # ld hl,1; ld a,l; or h -> ld a,1; or a (smaller), where HL is dead
            PeepholePattern(
                name="test_true_const_1",
                pattern=[("ld", "hl,1"), ("ld", "a,l"), ("or", "h")],
                replacement=[("ld", "a,1"), ("or", "a")],
                clobbers=hl,
            ),
            # ld hl,1; ld c,l -> ld c,1 (for shift count), where HL is dead
            PeepholePattern(
                name="ld_h1_cl",
                pattern=[("ld", "hl,1"), ("ld", "c,l")],
                replacement=[("ld", "c,1")],
                clobbers=hl,
            ),
            # ld hl,0; ld a,l; or h -> xor a (sets Z, clears A), where HL is dead
            PeepholePattern(
                name="test_false_const",
                pattern=[("ld", "hl,0"), ("ld", "a,l"), ("or", "h")],
                replacement=[("xor", "a")],
                clobbers=hl,
            ),
            # push hl; ld (addr),hl; pop hl -> ld (addr),hl
            # ld (addr),hl doesn't modify hl
            PeepholePattern(
                name="push_shld_pop",
                pattern=[("push", "hl"), ("ld", None), ("pop", "hl")],
                replacement=None,  # Keep middle only
                condition=lambda ops: ops[1][1].startswith("(") and ops[1][1].lower().endswith("),hl"),
            ),
            # push af; ld (addr),a; pop af -> ld (addr),a
            # Saving/restoring A around a store of A is pointless
            PeepholePattern(
                name="push_sta_pop",
                pattern=[("push", "af"), ("ld", None), ("pop", "af")],
                replacement=None,  # Keep middle only
                condition=lambda ops: ops[1][1].startswith("(") and ops[1][1].lower().endswith("),a"),
            ),
            # ld a,l; ld h,0; ld (addr),a -> ld a,l; ld (addr),a
            # ld h,0 is useless before the store if nothing - the store's own
            # address included, as in `ld (hl),a' - reads H.
            PeepholePattern(
                name="ld_al_h0_sta",
                pattern=[("ld", "a,l"), ("ld", "h,0"), ("ld", None)],
                replacement=None,  # Keep ld a,l and ld (addr),a
                condition=lambda ops: ops[2][1].startswith("(") and ops[2][1].lower().endswith("),a"),
                clobbers=frozenset("h"),
                dead_from=2,
            ),
            # ld l,a; ld h,0; ld (addr),a -> ld (addr),a
            # If we're just storing A, no need to extend to hl first
            PeepholePattern(
                name="ld_la_h0_sta",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("ld", None)],
                replacement=None,  # Keep only store
                condition=lambda ops: ops[2][1].startswith("(") and ops[2][1].lower().endswith("),a"),
                clobbers=hl,
                dead_from=2,
            ),
            # ld a,l; ld h,0; or h -> ld a,l; or a
            # h is 0, so or h is same as or a but or a is smaller
            PeepholePattern(
                name="ld_al_h0_or_h",
                pattern=[("ld", "a,l"), ("ld", "h,0"), ("or", "h")],
                replacement=[("ld", "a,l"), ("or", "a")],
                clobbers=frozenset("h"),
            ),
            # ld h,0; or h -> ld h,0; or a
            PeepholePattern(
                name="ld_h0_or_h",
                pattern=[("ld", "h,0"), ("or", "h")],
                replacement=[("ld", "h,0"), ("or", "a")],
            ),
            # Conditional jump followed by unconditional to same place
            # jp z,L; jp L -> jp L
            PeepholePattern(
                name="cond_uncond_same_z",
                pattern=[("jp", None), ("jp", None)],
                replacement=None,  # Keep second only
                condition=lambda ops: ops[0][1].lower().startswith("z,") and ops[0][1][2:] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_nz",
                pattern=[("jp", None), ("jp", None)],
                replacement=None,
                condition=lambda ops: ops[0][1].lower().startswith("nz,") and ops[0][1][3:] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_c",
                pattern=[("jp", None), ("jp", None)],
                replacement=None,
                condition=lambda ops: ops[0][1].lower().startswith("c,") and ops[0][1][2:] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_nc",
                pattern=[("jp", None), ("jp", None)],
                replacement=None,
                condition=lambda ops: ops[0][1].lower().startswith("nc,") and ops[0][1][3:] == ops[1][1],
            ),
            # ld a,(addr); cp y; jp z,z; ld a,(addr) -> ld a,(addr); cp y; jp z,z
            # A unchanged after cp/Jcond
            PeepholePattern(
                name="lda_cp_jz_lda_same",
                pattern=[("ld", None), ("cp", None), ("jp", None), ("ld", None)],
                replacement=None,  # Keep first 3 only
                condition=lambda ops: (ops[0][1].lower().startswith("a,(") and
                                       ops[2][1].lower().startswith("z,") and
                                       ops[0][1] == ops[3][1]),
            ),
            PeepholePattern(
                name="lda_cp_jnz_lda_same",
                pattern=[("ld", None), ("cp", None), ("jp", None), ("ld", None)],
                replacement=None,
                condition=lambda ops: (ops[0][1].lower().startswith("a,(") and
                                       ops[2][1].lower().startswith("nz,") and
                                       ops[0][1] == ops[3][1]),
            ),
            # ld a,(addr); or a; jp z,z; ld a,(addr) -> ld a,(addr); or a; jp z,z
            PeepholePattern(
                name="lda_or_jz_lda_same",
                pattern=[("ld", None), ("or", "a"), ("jp", None), ("ld", None)],
                replacement=None,
                condition=lambda ops: (ops[0][1].lower().startswith("a,(") and
                                       ops[2][1].lower().startswith("z,") and
                                       ops[0][1] == ops[3][1]),
            ),
            PeepholePattern(
                name="lda_or_jnz_lda_same",
                pattern=[("ld", None), ("or", "a"), ("jp", None), ("ld", None)],
                replacement=None,
                condition=lambda ops: (ops[0][1].lower().startswith("a,(") and
                                       ops[2][1].lower().startswith("nz,") and
                                       ops[0][1] == ops[3][1]),
            ),
        ]

    def optimize(self, asm_text: str) -> str:
        """Optimize Z80 assembly text."""
        lines = asm_text.split("\n")
        changed = True
        passes = 0
        max_passes = 10

        # Phase 1: Apply pattern-based and Z80-specific optimizations
        while changed and passes < max_passes:
            changed = False
            passes += 1
            lines, did_change = self._optimize_pass(lines)
            if did_change:
                changed = True
            lines, did_change = self._optimize_z80_pass(lines)
            if did_change:
                changed = True

        # Phase 2: Jump threading
        changed = True
        passes = 0
        while changed and passes < max_passes:
            changed = False
            passes += 1
            lines, did_change = self._jump_threading_pass(lines)
            if did_change:
                changed = True

        # Phase 3: Apply optimizations again, for what threading exposed
        lines, _ = self._optimize_pass(lines)
        lines, _ = self._optimize_z80_pass(lines)

        # Phase 4: Dead store elimination at procedure entry
        lines, _ = self._dead_store_elimination(lines)

        # Phase 5, last because it measures distances in bytes: relative
        # jumps and djnz where they reach.  Nothing after this may grow the
        # code.
        lines = self._convert_to_relative_jumps(lines)

        return "\n".join(lines)

    # ---- matching -----------------------------------------------------------

    def _window(self, lines: list[str], i: int, count: int, partial: bool = False
                ) -> tuple[list[int], list[Instr], list[str], int] | None:
        """The ``count`` instructions from line ``i`` on, if no label (which
        another path could enter by), data or other directive that emits or
        moves code comes first: their line numbers, the instructions, the
        comment and blank lines between them, and the line after the last.
        With ``partial``, as many as there are before such a line."""
        idxs: list[int] = []
        instrs: list[Instr] = []
        skipped: list[str] = []
        j = i
        while len(instrs) < count:
            if j >= len(lines):
                break
            line = lines[j]
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                skipped.append(line)
                j += 1
                continue
            if self._is_label_line(line):
                break
            label, op, operands = _split(line)
            if op is None or label is not None:
                break
            if op in TRANSPARENT and op not in _EQUATES:
                skipped.append(line)
                j += 1
                continue
            if op in DATA or op in BARRIERS or op in _EQUATES:
                break
            idxs.append(j)
            instrs.append((op, operands))
            j += 1
        if len(instrs) < count and not partial:
            return None
        return idxs, instrs, skipped, j

    def _prefix(self, lines: list[str], window: tuple[list[int], list[Instr], list[str], int] | None,
                count: int) -> tuple[list[int], list[Instr], list[str], int] | None:
        """The first ``count`` instructions of a partial window."""
        if window is None:
            return None
        idxs, instrs, _, _ = window
        if len(instrs) < count:
            return None
        first, last = idxs[0], idxs[count - 1]
        taken = set(idxs[:count])
        # The comment, blank and declaration lines among those taken.
        skipped = [lines[k] for k in range(first, last) if k not in taken]
        return idxs[:count], instrs[:count], skipped, last + 1

    def _clobbers_dead(self, code: _Code, pattern: PeepholePattern,
                       instrs: list[Instr], idxs: list[int]) -> bool:
        dead = pattern.clobbers(instrs) if callable(pattern.clobbers) else pattern.clobbers
        if not dead:
            return True
        start = idxs[pattern.dead_from] if pattern.dead_from is not None else idxs[-1] + 1
        return not code.live([start], dead)

    def _optimize_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """Apply pattern-based optimizations."""
        code = _Code(lines)
        longest_pattern = max(len(p.pattern) for p in self.patterns)
        result: list[str] = []
        changed = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines, comments, labels (also one with an instruction
            # after it, which the rewrite would lose), directives
            if not stripped or stripped.startswith(';') or self._is_label_line(line):
                result.append(line)
                i += 1
                continue

            parsed = self._parse_line(line)
            if parsed is None:
                result.append(line)
                i += 1
                continue

            # Special case: jp/jr to immediately following label
            if parsed[0] in ("jp", "jr") and "," not in parsed[1] and \
                    parsed[1].lower() not in ("(hl)", "(ix)", "(iy)"):
                target = parsed[1]
                # Look ahead for the target label (skip comments/empty lines)
                j = i + 1
                found_target = False
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith(";"):
                        j += 1
                        continue
                    # Check if this is a label line
                    if self._is_label_line(lines[j]):
                        label = next_line.split(":")[0].strip()
                        if label == target:
                            # JP to next label - remove the JP
                            self.stats["jump_to_next"] = self.stats.get("jump_to_next", 0) + 1
                            changed = True
                            found_target = True
                    break
                if found_target:
                    i += 1
                    continue

            # Try to match each pattern
            matched = False
            longest = self._window(lines, i, longest_pattern, partial=True)
            for pattern in self.patterns:
                if pattern.pattern[0][0] != parsed[0]:
                    continue
                window = self._prefix(lines, longest, len(pattern.pattern))
                if window is None:
                    continue
                instruction_lines, instrs, skipped, j = window

                if not self._matches_pattern(pattern, instrs):
                    continue
                if pattern.condition and not pattern.condition(instrs):
                    continue
                if not self._clobbers_dead(code, pattern, instrs, instruction_lines):
                    continue
                # Jumped to, a routine finds its caller's return address
                # where it looked for its own.  One of this text's that
                # takes it off the stack has to be called.
                if pattern.name == "tail_call" and code.moves_return(instrs[0][1]):
                    continue

                # Pattern matched!
                self.stats[pattern.name] = self.stats.get(pattern.name, 0) + 1
                changed = True
                matched = True

                # Preserve skipped comments/empty lines
                result.extend(skipped)

                # Apply replacement
                if pattern.replacement is not None:
                    for opcode, operands in pattern.replacement:
                        if operands:
                            result.append(f"\t{opcode} {operands}")
                        else:
                            result.append(f"\t{opcode}")
                elif pattern.name.startswith("cond_uncond"):
                    # Keep second instruction only
                    result.append(lines[instruction_lines[-1]])
                elif pattern.name in ("redundant_ld", "duplicate_ld", "ld_store_load_same", "sta_lda_same"):
                    # Keep first instruction only
                    result.append(lines[instruction_lines[0]])
                elif pattern.name in ("useless_extend_before_sub", "useless_extend_before_cp"):
                    # Keep last instruction only
                    result.append(lines[instruction_lines[-1]])
                elif pattern.name == "tail_call":
                    # call x; ret -> jp x
                    call_target = instrs[0][1]
                    result.append(f"\tjp {call_target}")
                elif pattern.name == "push_shld_pop":
                    # Keep middle only
                    result.append(lines[instruction_lines[1]])
                elif pattern.name == "push_sta_pop":
                    # Keep middle only
                    result.append(lines[instruction_lines[1]])
                elif pattern.name == "ld_al_h0_sta":
                    # Keep LD A,L and LD (addr),A
                    result.append(lines[instruction_lines[0]])
                    result.append(lines[instruction_lines[2]])
                elif pattern.name == "ld_la_h0_sta":
                    # Keep only store
                    result.append(lines[instruction_lines[2]])
                elif pattern.name in ("lda_cp_jz_lda_same", "lda_cp_jnz_lda_same",
                                      "lda_or_jz_lda_same", "lda_or_jnz_lda_same"):
                    # Keep first 3 instructions
                    result.append(lines[instruction_lines[0]])
                    result.append(lines[instruction_lines[1]])
                    result.append(lines[instruction_lines[2]])

                i = j
                break

            if not matched:
                result.append(line)
                i += 1

        return result, changed

    # ---- Z80-specific rewrites ----------------------------------------------

    def _optimize_z80_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """Apply Z80-specific inline optimizations."""
        code = _Code(lines)
        changed = False
        result: list[str] = []
        i = 0
        while i < len(lines):
            rewrite = self._z80_rewrite(lines, code, i)
            if rewrite is not None:
                new_lines, i, stat = rewrite
                result.extend(new_lines)
                changed = True
                self.stats[stat] = self.stats.get(stat, 0) + 1
                continue
            result.append(lines[i])
            i += 1
        return result, changed

    def _const_value(self, code: _Code, text: str) -> int | None:
        """The value of a numeric literal, or of a symbol this text sets
        with ``equ`` to one."""
        v = parse_number(text)
        if v is None:
            v = code.equ.get(text.strip())
        return v

    def _z80_rewrite(self, lines: list[str], code: _Code, i: int
                     ) -> tuple[list[str], int, str] | None:  # noqa: C901
        """The rewrite of the code starting at line ``i``: (new lines, the
        line to go on from, stat name), or None."""
        if self._is_label_line(lines[i]):
            return None
        parsed = self._parse_line(lines[i])
        if parsed is None:
            return None
        opcode, operands = parsed
        low = operands.lower()

        def dead(at: int, regs: frozenset[str] | set[str]) -> bool:
            return not code.live([at], regs)

        if opcode == "ld" and low.startswith("de,"):
            val = self._const_value(code, operands[3:])
            w = self._window(lines, i, 2)
            if w is not None and val is not None:
                _, ins, skipped, j = w
                op1, arg1 = ins[1][0], ins[1][1].lower()
                # ld de,1..3; add hl,de -> inc hl (repeated).  inc hl sets no
                # flags where add sets H, N and C, and DE keeps its value.
                if 1 <= val <= 3 and op1 == "add" and arg1 == "hl,de" and \
                        dead(j, {"d", "e", "fh", "fn", "fc"}):
                    return skipped + ["\tinc hl"] * val, j, "inc_hl_const"
                # ld de,2^k; call ??mul16 -> add hl,hl (k times).  Only as far
                # as 64: beyond, the shifts are longer than the call, and no
                # rewrite may lengthen code a relative jump already spans.
                # The routine may change any register but HL, and may leave
                # something in them; they have to be dead.
                if 2 <= val <= 64 and val & (val - 1) == 0 and op1 == "call" and \
                        arg1 in _MUL16 and dead(j, _NOT_HL):
                    return skipped + ["\tadd hl,hl"] * (val.bit_length() - 1), j, "mul_strength"
                # ld de,0; call ??subde -> (nothing): HL - 0 is HL, but the
                # routine sets the flags (and may use the other registers).
                if val == 0 and op1 == "call" and arg1 in _SUBDE and dead(j, _NOT_HL):
                    return skipped, j, "subde_zero"

        # ld a,(x); inc a; ld (x),a -> ld hl,x; inc (hl) where A and HL are
        # dead; for x = (hl) or (ix+d), inc (hl) or inc (ix+d) where A is.
        # The flags are the same: inc (hl) sets them as inc a does.
        if opcode == "ld" and low.startswith("a,"):
            src = classify(operands[2:])
            if src.kind in ("mem_abs", "mem_hl", "mem_idx"):
                w = self._window(lines, i, 3)
                if w is not None:
                    _, ins, skipped, j = w
                    (op1, arg1), (op2, arg2) = ins[1], ins[2]
                    dst = split_operands(arg2)
                    if op1 in ("inc", "dec") and arg1.lower() == "a" and op2 == "ld" and \
                            len(dst) == 2 and dst[1].lower() == "a" and \
                            _same_operand(dst[0], operands[2:]):
                        if src.kind == "mem_abs":
                            if dead(j, {"a", "h", "l"}):
                                addr = operands[2:].strip()[1:-1].strip()
                                return (skipped + [f"\tld hl,{addr}", f"\t{op1} (hl)"],
                                        j, f"{op1}_mem")
                        elif dead(j, {"a"}):
                            return skipped + [f"\t{op1} {src.text.strip()}"], j, f"{op1}_mem"

        # 8080-style 16-bit right shift to Z80 native:
        # or a / ld a,h / rra / ld h,a / ld a,l / rra / ld l,a -> srl h / rr l
        # (7 instructions -> 2).  A no longer ends up equal to L, and S, Z
        # and P/V are rr l's where they were or a's.
        if opcode == "or" and low == "a":
            w = self._window(lines, i, 7)
            if w is not None:
                _, ins, skipped, j = w
                shape = [(op, arg.lower()) for op, arg in ins[1:]]
                if shape == [("ld", "a,h"), ("rra", ""), ("ld", "h,a"), ("ld", "a,l"),
                             ("rra", ""), ("ld", "l,a")] and \
                        dead(j, {"a", "fs", "fz", "fp"}):
                    return skipped + ["\tsrl h", "\trr l"], j, "shr_z80"

        # push hl; ld hl,(addr); ex de,hl; pop hl -> ld de,(addr)
        # Z80 has direct ld de,(addr) which 8080 doesn't have
        if opcode == "push" and low == "hl":
            w = self._window(lines, i, 4)
            if w is not None:
                _, ins, skipped, j = w
                (op1, arg1), (op2, arg2), (op3, arg3) = ins[1], ins[2], ins[3]
                if op1 == "ld" and arg1.lower().startswith("hl,") and \
                        classify(arg1[3:]).kind == "mem_abs" and \
                        (op2, arg2.lower()) == ("ex", "de,hl") and (op3, arg3.lower()) == ("pop", "hl"):
                    return skipped + [f"\tld de,{arg1[3:]}"], j, "ld_de_addr"

        if opcode == "ld" and low.startswith("hl,") and not low.startswith("hl,("):
            const_text = operands[3:]
            val = self._const_value(code, const_text)
            # ld hl,const; ld r,l -> ld r,const, where both H and L are dead:
            # the whole ld hl,const goes.  The byte is L's, the constant's low
            # byte (`ld a,299' does not assemble), and it is taken only from
            # a constant whose value the text gives: a one-byte field holding
            # part of a relocatable or external address is something some
            # assemblers and linkers support and others do not.
            w = self._window(lines, i, 2)
            if w is not None and val is not None:
                _, ins, skipped, j = w
                op1, arg1 = ins[1]
                if op1 == "ld" and arg1.lower().endswith(",l"):
                    dest = arg1[:-2].strip().lower()
                    if dest in ("a", "b", "c", "d", "e") and dead(j, {"h", "l"}):
                        byte = const_text.strip() if 0 <= val <= 255 else str(val & 0xFF)
                        return skipped + [f"\tld {dest},{byte}"], j, "ld_via_hl"
            # ld hl,0; ld a,l; ld (addr),a -> xor a; ld (addr),a; ld hl,0, where
            # the flags xor a sets are dead, and the store does not address
            # through HL (which is 0 there in the original).
            if val == 0:
                w = self._window(lines, i, 3)
                if w is not None:
                    _, ins, skipped, j = w
                    (op1, arg1), (op2, arg2) = ins[1], ins[2]
                    dst = split_operands(arg2)
                    if (op1, arg1.lower()) == ("ld", "a,l") and op2 == "ld" and len(dst) == 2 and \
                            dst[1].lower() == "a" and \
                            classify(dst[0]).kind in ("mem_abs", "mem_idx", "mem_bc", "mem_de") and \
                            dead(j, FLAGS):
                        return (skipped + ["\txor a", f"\tld {dst[0]},a", f"\tld hl,{const_text}"],
                                j, "xor_a_store")

        # pop hl; push hl; ld hl,x -> ld hl,x
        if opcode == "pop" and low == "hl":
            w = self._window(lines, i, 3)
            if w is not None:
                idxs, ins, skipped, j = w
                if (ins[1][0], ins[1][1].lower()) == ("push", "hl") and ins[2][0] == "ld" and \
                        ins[2][1].lower().startswith("hl,") and effect(*ins[2]) is not UNKNOWN:
                    return skipped + [lines[idxs[2]]], j, "pop_push_ld"

        # ld hl,(addr1); push hl; ld hl,(addr2); ex de,hl; pop hl
        # -> ld de,(addr2); ld hl,(addr1)
        if opcode == "ld" and low.startswith("hl,") and classify(operands[3:]).kind == "mem_abs":
            w = self._window(lines, i, 5)
            if w is not None:
                _, ins, skipped, j = w
                if (ins[1][0], ins[1][1].lower()) == ("push", "hl") and ins[2][0] == "ld" and \
                        ins[2][1].lower().startswith("hl,") and \
                        classify(ins[2][1][3:]).kind == "mem_abs" and \
                        (ins[3][0], ins[3][1].lower()) == ("ex", "de,hl") and \
                        (ins[4][0], ins[4][1].lower()) == ("pop", "hl"):
                    return (skipped + [f"\tld de,{ins[2][1][3:]}", f"\tld hl,{operands[3:]}"],
                            j, "ld_de_nn")

        return None

    # ---- relative jumps -----------------------------------------------------

    def _convert_to_relative_jumps(self, lines: list[str]) -> list[str]:
        """Convert jp to jr, and dec b; jp/jr nz to djnz, where they reach.

        Distances are counted in bytes, and a conversion is made only where
        the displacement is known to fit.  Each conversion shortens the code,
        which can bring more jumps in range, so this repeats until none is
        left to convert."""
        while True:
            lines, changed = self._relative_jump_pass(lines)
            if not changed:
                return lines

    def _relative_jump_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        code = _Code(lines)
        result: list[str] = []
        changed = False
        i = 0
        while i < len(lines):
            line = lines[i]
            parsed = None if self._is_label_line(line) else self._parse_line(line)
            if parsed is None:
                result.append(line)
                i += 1
                continue
            opcode, operands = parsed

            # dec b; jp/jr nz,label -> djnz label.  dec b sets S, Z, H, P/V
            # and N and djnz sets none, so they must be dead where the loop
            # goes back and where it falls out.
            if opcode == "dec" and operands.lower() == "b":
                w = self._window(lines, i, 2)
                if w is not None:
                    idxs, ins, skipped, j = w
                    parts = split_operands(ins[1][1])
                    if ins[1][0] in ("jp", "jr") and len(parts) == 2 and parts[0].lower() == "nz":
                        t = code.target(parts[1])
                        if t is not None and code.reaches(idxs[0], idxs[1], t) and \
                                not code.live([t, idxs[1] + 1], FLAGS_NO_C):
                            result.extend(skipped)
                            result.append(f"\tdjnz {parts[1]}")
                            self.stats["djnz"] = self.stats.get("djnz", 0) + 1
                            changed = True
                            i = j
                            continue

            if opcode == "jp":
                parts = split_operands(operands)
                cond = parts[0].lower() if len(parts) == 2 else None
                target = parts[-1] if parts else ""
                if (cond is None or cond in JR_CONDITIONS) and len(parts) in (1, 2):
                    t = code.target(target)
                    if t is not None and code.reaches(i, i, t):
                        result.append(f"\tjr {cond},{target}" if cond else f"\tjr {target}")
                        self.stats["jr_convert"] = self.stats.get("jr_convert", 0) + 1
                        changed = True
                        i += 1
                        continue

            result.append(line)
            i += 1
        return result, changed

    # ---- jump threading -----------------------------------------------------

    def _jump_threading_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """
        Jump threading optimization.

        If a jump targets a label whose only content is another unconditional jump,
        thread through to the final destination.
        """
        changed = False
        code = _Code(lines)

        # Build map of label -> (line index, first instruction at or after it)
        label_info: dict[str, tuple[int, Instr | None]] = {}
        for i, line in enumerate(lines):
            if self._is_label_line(line):
                label, op, operands = _split(line)
                if label is None or op in _EQUATES:
                    continue
                first_instr: Instr | None = None
                if op is not None:
                    # An instruction on the label's own line comes first.
                    first_instr = self._parse_line(line)
                else:
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line or next_line.startswith(";"):
                            continue
                        if self._is_label_line(lines[j]):
                            break
                        first_instr = self._parse_line(lines[j])
                        break
                label_info[label] = (i, first_instr)

        def plain_jump(ins: Instr | None) -> str | None:
            if ins and ins[0] in ("jp", "jr") and "," not in ins[1] and \
                    ins[1].lower() not in ("(hl)", "(ix)", "(iy)"):
                return ins[1].strip()
            return None

        # Build map of label -> final destination
        label_target: dict[str, str] = {}
        for label, (_, first_instr) in label_info.items():
            target = plain_jump(first_instr)
            if target is None:
                continue
            # Follow the chain
            visited = {label}
            while target in label_info and target not in visited:
                visited.add(target)
                nxt = plain_jump(label_info[target][1])
                if nxt is None:
                    break
                target = nxt
            if target != label:
                label_target[label] = target

        # Rewrite jumps to use final destinations
        result: list[str] = []
        for i, line in enumerate(lines):
            parsed = None if self._is_label_line(line) else self._parse_line(line)
            if parsed and parsed[0] in ("jp", "jr") and "," not in parsed[1] and \
                    parsed[1].strip() in label_target:
                new_target = label_target[parsed[1].strip()]
                if parsed[0] == "jp":
                    result.append(f"\tjp {new_target}")
                else:
                    # A jr has to reach its new target; if it may not, it
                    # stays as it is (making it a jp would lengthen code
                    # other relative jumps already span).
                    t = code.target(new_target)
                    if t is None or not code.reaches(i, i, t):
                        result.append(line)
                        continue
                    result.append(f"\tjr {new_target}")
                changed = True
                self.stats["jump_thread"] = self.stats.get("jump_thread", 0) + 1
            elif parsed and parsed[0] == "dw" and parsed[1].strip() in label_target:
                # Thread dw references
                result.append(f"\tdw {label_target[parsed[1].strip()]}")
                changed = True
                self.stats["dw_thread"] = self.stats.get("dw_thread", 0) + 1
            else:
                result.append(line)

        # What names each label: any mention of it outside its definition,
        # in any instruction or directive, counts, and so does `NAME::',
        # which exports it.
        refs: dict[str, int] = {}
        for line in result:
            name = _exported(line)
            if name:
                refs[name.lower()] = refs.get(name.lower(), 0) + 1
            label, op, operands = _split(line)
            if op is None:
                continue
            for name in _IDENT.findall(operands):
                refs[name.lower()] = refs.get(name.lower(), 0) + 1

        # Remove unreferenced labels that just jump, where nothing falls into them
        final_result: list[str] = []
        i = 0
        while i < len(result):
            line = result[i]

            if self._is_label_line(line):
                label, op, _ = _split(line)
                if label is not None and label in label_target and refs.get(label.lower(), 0) == 0 and \
                        not self._falls_through(final_result):
                    changed = True
                    self.stats["dead_label_removed"] = self.stats.get("dead_label_removed", 0) + 1
                    i += 1
                    if op is not None:
                        # The jump was on the label's own line.
                        continue
                    # Skip the jump instruction too
                    while i < len(result):
                        next_line = result[i].strip()
                        if not next_line or next_line.startswith(";"):
                            i += 1
                            continue
                        next_parsed = self._parse_line(next_line)
                        if next_parsed and next_parsed[0] in ("jp", "jr"):
                            i += 1
                            break
                        break
                    continue

            final_result.append(line)
            i += 1

        return final_result, changed

    def _falls_through(self, done: list[str]) -> bool:
        """Can control run off the end of ``done`` into what follows?"""
        for j in range(len(done) - 1, -1, -1):
            prev = done[j].strip()
            if not prev or prev.startswith(";"):
                continue
            if self._is_label_line(done[j]) and _split(done[j])[1] is None:
                return True
            parsed = self._parse_line(done[j])
            if parsed is None:
                return True
            op, operands = parsed
            if op in ("jp", "jr") and "," not in operands:
                return False
            if op in ("ret", "reti", "retn") and not operands:
                return False
            return True
        return True

    # ---- dead stores --------------------------------------------------------

    def _dead_store_elimination(self, lines: list[str]) -> tuple[list[str], bool]:
        """
        Eliminate dead stores at procedure entry.

        Pattern: A procedure stores a register parameter to memory at entry,
        but uses the register directly without ever loading from that memory.

        Two things make a store here safe to drop, and both are checked:

        * The label really is a procedure entry.  A compiler-internal label
          (``??CMP0003:`` and the like) is a branch target in the middle of an
          expression, and the store that follows it is an ordinary assignment
          which happens to land right after a join point - most often
          ``var = (a = b)``, where the two arms of the comparison meet at the
          label before the result is stored.

        * Nothing else in the module reads the location.  Scanning only to the
          end of the procedure is enough for a parameter slot, which no one
          else can name, and wrong for anything at module scope: another
          procedure, declared later, reads it perfectly legally.  A location
          whose address is taken, or which the module exports, is off limits
          for the same reason.
        """
        result: list[str] = []
        changed = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Look for procedure entry (label followed by ld (addr),a)
            if self._is_label_line(line) and not stripped.startswith("??") and \
                    _split(line)[1] is None:
                if i + 1 < len(lines) and not self._is_label_line(lines[i + 1]):
                    parsed = self._parse_line(lines[i + 1])
                    # Check for ld (addr),a pattern
                    if (parsed and parsed[0] == "ld" and
                            parsed[1].startswith("(") and parsed[1].lower().endswith("),a") and
                            classify(parsed[1][:-2]).kind == "mem_abs"):
                        addr = parsed[1][1:-3]  # Extract addr from (addr),a

                        # Check whether anything anywhere in the module reads
                        # the location, exports it, or takes its address.  The
                        # store is only dead if nothing does.
                        addr_loaded = self._addr_is_live(lines, addr, i + 1)

                        if not addr_loaded:
                            result.append(line)  # Keep the label
                            i += 2  # Skip the store instruction
                            changed = True
                            self.stats["dead_store_elim"] = self.stats.get("dead_store_elim", 0) + 1
                            continue

            result.append(line)
            i += 1

        return result, changed

    def _addr_is_live(self, lines: list[str], addr: str, store_index: int) -> bool:
        """Does anything outside ``lines[store_index]`` need ``addr``?

        Conservative by construction: a name that is read, exported, or used
        anywhere as a value rather than as a store destination counts as live.
        """
        target = addr.strip().lower()
        if not target:
            return True
        paren = "(" + target + ")"
        # A sixteen-bit load from the byte before reads this one too.
        m = re.fullmatch(r"(.*?)\+(\d+)", target)
        below: set[str] = set()
        if m and int(m.group(2)) >= 1:
            k = int(m.group(2)) - 1
            below = {f"({m.group(1)}+{k})"} | ({f"({m.group(1)})"} if k == 0 else set())
        for j, raw in enumerate(lines):
            if j == store_index:
                continue
            text = raw.strip().lower()
            if not text or text.startswith(";"):
                continue
            if below and any(b in text for b in below):
                parsed = self._parse_line(text)
                if parsed and parsed[0] == "ld":
                    parts = split_operands(parsed[1])
                    if len(parts) == 2 and parts[1] in below and \
                            classify(parts[0]).kind == "r16":
                        return True
            if target not in text:
                continue
            parsed = self._parse_line(text)
            if not parsed:
                # A label definition or a directive. `public NAME' hands the
                # name to another module, which may read it, and `ALIAS equ
                # NAME' or `db low(NAME)' names it too; the line that defines
                # the location is no use of it.
                label, op, operand = _split(raw)
                if label is not None and label.lower() == target and \
                        target not in operand.lower():
                    continue
                if op is None:
                    continue
                return True
            op, operand = parsed[0], parsed[1] if len(parsed) > 1 else ""
            if op in ("public", "global", "extrn", "external"):
                return True
            if op == "ld" and operand.startswith(paren):
                # A store to it. Not a read.
                continue
            # Anything else that names it - a load, an address taken with
            # `ld hl,NAME', a `dw NAME' in a table - keeps it alive.
            return True
        return False

    # ---- lines --------------------------------------------------------------

    def _is_label_line(self, line: str) -> bool:
        """Check if a raw (unstripped) line is a label definition."""
        return (":" in line and
                not line.startswith(("\t", " ")) and
                not line.startswith(";"))

    def _parse_line(self, line: str) -> tuple[str, str] | None:
        """Parse a Z80 assembly line into (opcode, operands).

        A label in column 1 is dropped (the instruction after it, if any, is
        returned), and so is a trailing comment.  Directives give None, except
        ``dw``, whose operand jump threading follows."""
        label, opcode, operands = _split(line)
        if opcode is None:
            return None
        if opcode in _EQUATES and label is not None:
            return None
        if opcode in TRANSPARENT | DATA | BARRIERS and opcode != "dw":
            return None
        return (opcode, operands)

    def _parse_const(self, s: str) -> int | None:
        """Parse an assembly constant value. Returns None if not a constant."""
        return parse_number(s)

    def _matches_pattern(
        self, pattern: PeepholePattern, instructions: list[tuple[str, str]]
    ) -> bool:
        """Check if instructions match the pattern."""
        if len(instructions) != len(pattern.pattern):
            return False

        for (pat_op, pat_operands), (inst_op, inst_operands) in zip(
            pattern.pattern, instructions
        ):
            if pat_op != inst_op:
                return False

            if pat_operands is not None:
                if "*" in pat_operands:
                    # Wildcard match
                    pat_re = ".*".join(re.escape(p) for p in pat_operands.split("*"))
                    if not re.fullmatch(pat_re, inst_operands, re.IGNORECASE):
                        return False
                elif pat_operands.lower() != inst_operands.lower():
                    return False

        return True


def optimize(asm_text: str) -> str:
    """Optimize Z80 assembly code.

    This is the main entry point for the optimizer.
    Pass Z80 assembly text (using ld, jp, jr, etc. mnemonics)
    and receive optimized Z80 assembly back.
    """
    optimizer = PeepholeOptimizer()
    return optimizer.optimize(asm_text)
