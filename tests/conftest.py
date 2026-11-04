"""Every call of ``PeepholeOptimizer.optimize`` in the tests is also run on
the same text with ``;#`` debug markers added (docs: um80_and_friends
docs/debug_info.md).  The optimizer must keep each marker, in order, and
must write the same instructions as without the markers: a debug build of a
compiler gives the same code as a build without debug markers.
"""

import re

import pytest

from upeepz80 import peephole

_KINDS = r"(?:file|line|inline|endinline|func|endfunc|noopt)(?![A-Za-z0-9_])"
_MARKER = re.compile(r"^[ \t]*;[ \t]*#" + _KINDS, re.IGNORECASE)
_HAS_MARKER = re.compile(r"^[ \t]*;[ \t]*#" + _KINDS, re.MULTILINE | re.IGNORECASE)
_LABEL = re.compile(r"^[ \t]*[A-Za-z_@?$.][A-Za-z0-9_@?$.]*::?")
# first words that are not instructions: a compiler writes no marker in data
_DIRECTIVES = frozenset((
    "db", "dw", "dd", "ds", "defb", "defw", "defs", "defm", "defl", "equ",
    "public", "extrn", "extern", "global", "cseg", "dseg", "aseg", "common",
    "org", "end", "name", "title", ".z80", ".8080", "if", "else", "endif",
    "macro", "endm", "rept", "irp", "irpc", "assume", ".phase", ".dephase",
    "area", "module", "include"))


def add_markers(text: str) -> str:
    """The text with ``;#line 1 N`` in front of each instruction line (N
    counts from 1, so each marker is different), and one at the end."""
    out = []
    n = 0
    for line in text.split("\n"):
        body = _LABEL.sub("", line.split(";", 1)[0], count=1).strip()
        word = body.split(None, 1)[0].lower() if body else ""
        if word and word not in _DIRECTIVES and not _MARKER.match(line):
            n += 1
            out.append(f";#line 1 {n}")
        out.append(line)
    out.append(f";#line 1 {n + 1}")
    return "\n".join(out)


def markers_of(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if _MARKER.match(ln)]


def without_markers(text: str) -> str:
    return "\n".join(ln for ln in text.split("\n") if not _MARKER.match(ln))


@pytest.fixture(autouse=True)
def marker_check(monkeypatch):
    original = peephole.PeepholeOptimizer.optimize
    seen = []

    def checked(self, asm_text):
        result = original(self, asm_text)
        if not isinstance(asm_text, str) or _HAS_MARKER.search(asm_text):
            return result
        marked = add_markers(asm_text)
        other = peephole.PeepholeOptimizer(disable=self.disable)
        got = original(other, marked)
        seen.append((asm_text, result, marked, got))
        return result

    monkeypatch.setattr(peephole.PeepholeOptimizer, "optimize", checked)
    yield
    for asm_text, result, marked, got in seen:
        assert without_markers(got) == result, (
            "markers changed the instructions:\n" + marked)
        assert markers_of(got) == markers_of(marked), (
            "markers lost or moved:\n" + marked + "\n--- got ---\n" + got)
