"""
Peephole Optimizer for Z80.

Performs pattern-based optimizations on generated Z80 assembly code.
This runs after code generation to clean up inefficient sequences.

This module expects pure Z80 mnemonics as input (ld, jp, jr, etc.)
and produces optimized Z80 assembly as output. All output uses
lowercase mnemonics and register names.

For compilers that generate 8080 mnemonics, use upeep80 instead.
"""

import re
from dataclasses import dataclass
from typing import Callable


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
            # Zero A: ld a,0 -> xor a (smaller, faster)
            PeepholePattern(
                name="zero_a_ld",
                pattern=[("ld", "a,0")],
                replacement=[("xor", "a")],
            ),
            # Compare to zero: cp 0 -> or a (sets Z flag, smaller)
            PeepholePattern(
                name="cp_zero",
                pattern=[("cp", "0")],
                replacement=[("or", "a")],
            ),
            # Redundant duplicate ld: ld x,y; ld x,y -> ld x,y
            PeepholePattern(
                name="duplicate_ld",
                pattern=[("ld", None), ("ld", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1].lower() == ops[1][1].lower(),
            ),
            # ld a,a -> (nothing, useless)
            PeepholePattern(
                name="ld_a_a",
                pattern=[("ld", "a,a")],
                replacement=[],
            ),
            # ld b,b, ld c,c, etc. -> (nothing)
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
            # inc a; dec a -> (nothing)
            PeepholePattern(
                name="inc_dec_a",
                pattern=[("inc", "a"), ("dec", "a")],
                replacement=[],
            ),
            # dec a; inc a -> (nothing)
            PeepholePattern(
                name="dec_inc_a",
                pattern=[("dec", "a"), ("inc", "a")],
                replacement=[],
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
            # ccf; ccf -> (nothing) - complement carry twice
            PeepholePattern(
                name="double_ccf",
                pattern=[("ccf", ""), ("ccf", "")],
                replacement=[],
            ),
            # cpl; cpl -> (nothing) - complement A twice
            PeepholePattern(
                name="double_cpl",
                pattern=[("cpl", ""), ("cpl", "")],
                replacement=[],
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
            # call x; ret -> jp x (tail call optimization)
            PeepholePattern(
                name="tail_call",
                pattern=[("call", None), ("ret", "")],
                replacement=None,  # Replaced specially
                condition=lambda ops: True,
            ),
            # ret; ret -> ret (unreachable code)
            PeepholePattern(
                name="double_ret",
                pattern=[("ret", ""), ("ret", "")],
                replacement=[("ret", "")],
            ),
            # ld a,(hl); ld e,a -> ld e,(hl)
            PeepholePattern(
                name="ld_a_hl_ld_ea",
                pattern=[("ld", "a,(hl)"), ("ld", "e,a")],
                replacement=[("ld", "e,(hl)")],
            ),
            # ld a,(hl); ld d,a -> ld d,(hl)
            PeepholePattern(
                name="ld_a_hl_ld_da",
                pattern=[("ld", "a,(hl)"), ("ld", "d,a")],
                replacement=[("ld", "d,(hl)")],
            ),
            # ld a,(hl); ld c,a -> ld c,(hl)
            PeepholePattern(
                name="ld_a_hl_ld_ca",
                pattern=[("ld", "a,(hl)"), ("ld", "c,a")],
                replacement=[("ld", "c,(hl)")],
            ),
            # ld a,(hl); ld b,a -> ld b,(hl)
            PeepholePattern(
                name="ld_a_hl_ld_ba",
                pattern=[("ld", "a,(hl)"), ("ld", "b,a")],
                replacement=[("ld", "b,(hl)")],
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
            # and 0ffh -> or a (same effect, smaller)
            PeepholePattern(
                name="and_ff",
                pattern=[("and", "0ffh")],
                replacement=[("or", "a")],
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
            # ld h,0; ld d,h; ld e,l -> ld d,0; ld e,l
            # d = h = 0, so just load d directly with 0
            PeepholePattern(
                name="ld_h0_dh_el",
                pattern=[("ld", "h,0"), ("ld", "d,h"), ("ld", "e,l")],
                replacement=[("ld", "d,0"), ("ld", "e,l")],
            ),
            # Wasteful byte extension before byte op: ld l,a; ld h,0; sub x -> sub x
            # (Also for cp, and, or, xor, add byte ops)
            PeepholePattern(
                name="useless_extend_before_sub",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("sub", None)],
                replacement=None,  # Keep last only
                condition=lambda ops: True,  # Always apply
            ),
            PeepholePattern(
                name="useless_extend_before_cp",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("cp", None)],
                replacement=None,  # Keep last only
                condition=lambda ops: True,
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
            # ld hl,0ffffh; ld a,l; or h -> ld hl,0ffffh; or a
            # Since 0xFFFF is always true
            PeepholePattern(
                name="test_true_const",
                pattern=[("ld", "hl,0ffffh"), ("ld", "a,l"), ("or", "h")],
                replacement=[("ld", "hl,0ffffh"), ("or", "a")],
            ),
            # ld hl,1; ld a,l; or h -> ld a,1; or a (smaller)
            PeepholePattern(
                name="test_true_const_1",
                pattern=[("ld", "hl,1"), ("ld", "a,l"), ("or", "h")],
                replacement=[("ld", "a,1"), ("or", "a")],
            ),
            # ld hl,1; ld c,l -> ld c,1 (for shift count)
            PeepholePattern(
                name="ld_h1_cl",
                pattern=[("ld", "hl,1"), ("ld", "c,l")],
                replacement=[("ld", "c,1")],
            ),
            # ld hl,0; ld a,l; or h -> xor a (sets Z, clears A)
            PeepholePattern(
                name="test_false_const",
                pattern=[("ld", "hl,0"), ("ld", "a,l"), ("or", "h")],
                replacement=[("xor", "a")],
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
            # mvi h,0 is useless before store
            PeepholePattern(
                name="ld_al_h0_sta",
                pattern=[("ld", "a,l"), ("ld", "h,0"), ("ld", None)],
                replacement=None,  # Keep ld a,l and ld (addr),a
                condition=lambda ops: ops[2][1].startswith("(") and ops[2][1].lower().endswith("),a"),
            ),
            # ld l,a; ld h,0; ld (addr),a -> ld (addr),a
            # If we're just storing A, no need to extend to hl first
            PeepholePattern(
                name="ld_la_h0_sta",
                pattern=[("ld", "l,a"), ("ld", "h,0"), ("ld", None)],
                replacement=None,  # Keep only store
                condition=lambda ops: ops[2][1].startswith("(") and ops[2][1].lower().endswith("),a"),
            ),
            # ld a,l; ld h,0; or h -> ld a,l; or a
            # h is 0, so or h is same as or a but or a is smaller
            PeepholePattern(
                name="ld_al_h0_or_h",
                pattern=[("ld", "a,l"), ("ld", "h,0"), ("or", "h")],
                replacement=[("ld", "a,l"), ("or", "a")],
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

        # Phase 1: Apply Z80 patterns
        while changed and passes < max_passes:
            changed = False
            passes += 1
            lines, did_change = self._optimize_pass(lines)
            if did_change:
                changed = True

        # Phase 2: Z80-specific optimizations (inline patterns)
        changed = True
        passes = 0
        while changed and passes < max_passes:
            changed = False
            passes += 1
            lines, did_change = self._optimize_z80_pass(lines)
            if did_change:
                changed = True

        # Phase 3: Jump threading
        changed = True
        passes = 0
        while changed and passes < max_passes:
            changed = False
            passes += 1
            lines, did_change = self._jump_threading_pass(lines)
            if did_change:
                changed = True

        # Phase 4: Convert long jumps to relative jumps where possible
        lines = self._convert_to_relative_jumps(lines)

        # Phase 5: Apply Z80-specific patterns again (for DJNZ after JR conversion)
        lines, _ = self._optimize_z80_pass(lines)

        # Phase 6: Dead store elimination at procedure entry
        lines, _ = self._dead_store_elimination(lines)

        return "\n".join(lines)

    def _optimize_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """Apply pattern-based optimizations."""
        result: list[str] = []
        changed = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines, comments, labels, directives
            if not stripped or stripped.startswith(';') or stripped.endswith(':'):
                result.append(line)
                i += 1
                continue

            if stripped.startswith('.') or stripped.lower().startswith(('org', 'equ', 'db', 'dw', 'ds')):
                result.append(line)
                i += 1
                continue

            # Special case: jp/jr to immediately following label
            parsed = self._parse_line(lines[i])
            if parsed and parsed[0] in ("jp", "jr") and "," not in parsed[1] and parsed[1].lower() != "(hl)":
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
                    if ":" in next_line and not next_line.startswith("\t"):
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
            for pattern in self.patterns:
                match_len = len(pattern.pattern)
                if i + match_len > len(lines):
                    continue

                # Extract instructions for pattern matching
                instrs: list[tuple[str, str]] = []
                instruction_lines: list[int] = []
                skip_indices: list[int] = []
                valid = True

                j = i
                instr_count = 0
                while instr_count < match_len and j < len(lines):
                    instr_line = lines[j].strip()
                    parsed = self._parse_line(lines[j])
                    if parsed is None:
                        # Check for label - breaks pattern matching
                        if instr_line and ':' in instr_line and not instr_line.startswith(';'):
                            valid = False
                            break
                        skip_indices.append(j - i)
                        j += 1
                        continue
                    instrs.append(parsed)
                    instruction_lines.append(j)
                    instr_count += 1
                    j += 1

                if not valid or len(instrs) != match_len:
                    continue

                # Check if pattern matches
                if self._matches_pattern(pattern, instrs):
                    # Apply condition if present
                    if pattern.condition and not pattern.condition(instrs):
                        continue

                    # Pattern matched!
                    self.stats[pattern.name] = self.stats.get(pattern.name, 0) + 1
                    changed = True
                    matched = True

                    # Preserve skipped comments/empty lines
                    for offset in skip_indices:
                        result.append(lines[i + offset])

                    # Apply replacement
                    if pattern.replacement is not None:
                        for opcode, operands in pattern.replacement:
                            if operands:
                                result.append(f"    {opcode} {operands}")
                            else:
                                result.append(f"    {opcode}")
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
                        result.append(f"    jp {call_target}")
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

    def _optimize_z80_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """Apply Z80-specific inline optimizations."""
        changed = False
        result: list[str] = []
        i = 0

        # Build label_lines map for range checking
        label_lines: dict[str, int] = {}
        for line_num, line in enumerate(lines):
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("\t"):
                label = stripped.split(":")[0].strip()
                label_lines[label] = line_num

        while i < len(lines):
            line = lines[i].strip()
            parsed = self._parse_line(line)

            if parsed:
                opcode, operands = parsed

                # ld a,0 -> xor a (1 byte vs 2)
                if opcode == "ld" and operands.lower() == "a,0":
                    result.append("\txor a")
                    changed = True
                    self.stats["xor_a"] = self.stats.get("xor_a", 0) + 1
                    i += 1
                    continue

                # ld de,1/2/3; add hl,de -> inc hl (repeated)
                # Saves 3/2/1 bytes respectively
                if opcode == "ld" and operands.lower().startswith("de,"):
                    const_str = operands[3:].strip()
                    const_val = self._parse_const(const_str)
                    if const_val is not None and 1 <= const_val <= 3:
                        if i + 1 < len(lines):
                            p1 = self._parse_line(lines[i + 1].strip())
                            if p1 and p1[0] == "add" and p1[1].lower() == "hl,de":
                                for _ in range(const_val):
                                    result.append("\tinc hl")
                                changed = True
                                self.stats["inc_hl_const"] = self.stats.get("inc_hl_const", 0) + 1
                                i += 2
                                continue

                # ld de,{power-of-2}; call ??mul16 -> add hl,hl (repeated)
                # Strength reduction for multiply by power of 2
                if opcode == "ld" and operands.lower().startswith("de,"):
                    const_str = operands[3:].strip()
                    const_val = self._parse_const(const_str)
                    if const_val is not None and const_val > 1:
                        # Check if power of 2: x & (x-1) == 0
                        if (const_val & (const_val - 1)) == 0:
                            if i + 1 < len(lines):
                                p1 = self._parse_line(lines[i + 1].strip())
                                if p1 and p1[0] == "call" and p1[1].lower() in ("??mul16", "@mul16"):
                                    # Count shifts needed (log2)
                                    shift_count = 0
                                    temp = const_val
                                    while temp > 1:
                                        temp >>= 1
                                        shift_count += 1
                                    for _ in range(shift_count):
                                        result.append("\tadd hl,hl")
                                    changed = True
                                    self.stats["mul_strength"] = self.stats.get("mul_strength", 0) + 1
                                    i += 2
                                    continue

                # ld a,(addr); inc a; ld (addr),a -> ld hl,addr; inc (hl)
                if opcode == "ld" and operands.lower().startswith("a,(") and operands.endswith(")"):
                    addr = operands[3:-1]  # Extract address
                    if i + 2 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        p2 = self._parse_line(lines[i + 2].strip())
                        if (p1 and p1[0] == "inc" and p1[1].lower() == "a" and
                            p2 and p2[0] == "ld" and p2[1].lower() == f"({addr.lower()}),a"):
                            result.append(f"\tld hl,{addr}")
                            result.append("\tinc (hl)")
                            changed = True
                            self.stats["inc_mem"] = self.stats.get("inc_mem", 0) + 1
                            i += 3
                            continue
                        # Also check for dec a
                        if (p1 and p1[0] == "dec" and p1[1].lower() == "a" and
                            p2 and p2[0] == "ld" and p2[1].lower() == f"({addr.lower()}),a"):
                            result.append(f"\tld hl,{addr}")
                            result.append("\tdec (hl)")
                            changed = True
                            self.stats["dec_mem"] = self.stats.get("dec_mem", 0) + 1
                            i += 3
                            continue

                # ex de,hl; ex de,hl -> (nothing)
                if opcode == "ex" and operands.lower() == "de,hl" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "ex" and next_parsed[1].lower() == "de,hl":
                        changed = True
                        self.stats["double_ex"] = self.stats.get("double_ex", 0) + 1
                        i += 2
                        continue

                # inc hl; dec hl -> (nothing)
                if opcode == "inc" and operands.lower() == "hl" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "dec" and next_parsed[1].lower() == "hl":
                        changed = True
                        self.stats["inc_dec_hl"] = self.stats.get("inc_dec_hl", 0) + 1
                        i += 2
                        continue

                # dec hl; inc hl -> (nothing)
                if opcode == "dec" and operands.lower() == "hl" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "inc" and next_parsed[1].lower() == "hl":
                        changed = True
                        self.stats["dec_inc_hl"] = self.stats.get("dec_inc_hl", 0) + 1
                        i += 2
                        continue

                # ld (addr),hl; ld hl,(addr) -> ld (addr),hl (same address)
                if opcode == "ld" and operands.startswith("(") and operands.lower().endswith("),hl"):
                    addr = operands[1:-4]
                    if i + 1 < len(lines):
                        next_parsed = self._parse_line(lines[i + 1].strip())
                        if next_parsed and next_parsed[0] == "ld" and next_parsed[1].lower() == f"hl,({addr.lower()})":
                            result.append(lines[i])
                            changed = True
                            self.stats["ld_hl_same"] = self.stats.get("ld_hl_same", 0) + 1
                            i += 2
                            continue

                # dec b; jr/jp nz,label -> djnz label
                if opcode == "dec" and operands.lower() == "b" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] in ("jr", "jp") and next_parsed[1].lower().startswith("nz,"):
                        target = next_parsed[1][3:]  # Remove "NZ,"
                        if target in label_lines:
                            distance = label_lines[target] - i
                            if -50 < distance < 50:
                                result.append(f"\tdjnz {target}")
                                changed = True
                                self.stats["djnz"] = self.stats.get("djnz", 0) + 1
                                i += 2
                                continue

                # 8080-style 16-bit right shift to Z80 native:
                # or a / ld a,h / rra / ld h,a / ld a,l / rra / ld l,a -> srl h / rr l
                # (7 instructions -> 2 instructions)
                if opcode == "or" and operands.lower() == "a" and i + 6 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    p2 = self._parse_line(lines[i + 2].strip())
                    p3 = self._parse_line(lines[i + 3].strip())
                    p4 = self._parse_line(lines[i + 4].strip())
                    p5 = self._parse_line(lines[i + 5].strip())
                    p6 = self._parse_line(lines[i + 6].strip())
                    if (p1 and p1[0] == "ld" and p1[1].lower() == "a,h" and
                        p2 and p2[0] == "rra" and
                        p3 and p3[0] == "ld" and p3[1].lower() == "h,a" and
                        p4 and p4[0] == "ld" and p4[1].lower() == "a,l" and
                        p5 and p5[0] == "rra" and
                        p6 and p6[0] == "ld" and p6[1].lower() == "l,a"):
                        result.append("\tsrl h")
                        result.append("\trr l")
                        changed = True
                        self.stats["shr_z80"] = self.stats.get("shr_z80", 0) + 1
                        i += 7
                        continue

                # Compare to zero via ??subde: ld de,0 / call ??subde -> (remove, just test)
                # The subtraction of 0 is pointless before a zero test
                if opcode == "ld" and operands.lower() == "de,0" and i + 1 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    if p1 and p1[0] == "call" and p1[1].lower() in ("??subde", "@subde"):
                        # Skip both instructions - the value in HL is unchanged
                        changed = True
                        self.stats["subde_zero"] = self.stats.get("subde_zero", 0) + 1
                        i += 2
                        continue

                # push hl; ld hl,(addr); ex de,hl; pop hl -> ld de,(addr)
                # Z80 has direct ld de,(addr) which 8080 doesn't have
                if opcode == "push" and operands.lower() == "hl" and i + 3 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    p2 = self._parse_line(lines[i + 2].strip())
                    p3 = self._parse_line(lines[i + 3].strip())
                    if (p1 and p1[0] == "ld" and p1[1].lower().startswith("hl,(") and p1[1].endswith(")") and
                        p2 and p2[0] == "ex" and p2[1].lower() == "de,hl" and
                        p3 and p3[0] == "pop" and p3[1].lower() == "hl"):
                        addr = p1[1][3:]  # Get (addr) including parens
                        result.append(f"\tld de,{addr}")
                        changed = True
                        self.stats["ld_de_addr"] = self.stats.get("ld_de_addr", 0) + 1
                        i += 4
                        continue

                # push af; ld (addr),a; pop af -> ld (addr),a
                if opcode == "push" and operands.lower() == "af" and i + 2 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    p2 = self._parse_line(lines[i + 2].strip())
                    if (p1 and p1[0] == "ld" and p1[1].startswith("(") and p1[1].lower().endswith("),a") and
                        p2 and p2[0] == "pop" and p2[1].lower() == "af"):
                        result.append(lines[i + 1])  # Keep only ld (addr),a
                        changed = True
                        self.stats["push_sta_pop"] = self.stats.get("push_sta_pop", 0) + 1
                        i += 3
                        continue

                # ld hl,const; ld r,l -> ld r,const
                if opcode == "ld" and operands.lower().startswith("hl,") and not operands.lower().startswith("hl,("):
                    const_val = operands[3:]
                    if i + 1 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        if p1 and p1[0] == "ld" and p1[1].lower().endswith(",l"):
                            dest_reg = p1[1][:-2]  # Get destination register
                            if dest_reg.lower() in ("a", "b", "c", "d", "e"):
                                result.append(f"\tld {dest_reg.lower()},{const_val}")
                                changed = True
                                self.stats["ld_via_hl"] = self.stats.get("ld_via_hl", 0) + 1
                                i += 2
                                continue

                # pop hl; push hl; ld hl,x -> ld hl,x
                if opcode == "pop" and operands.lower() == "hl" and i + 2 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    p2 = self._parse_line(lines[i + 2].strip())
                    if (p1 and p1[0] == "push" and p1[1].lower() == "hl" and
                        p2 and p2[0] == "ld" and p2[1].lower().startswith("hl,")):
                        result.append(lines[i + 2])  # Keep only ld hl,x
                        changed = True
                        self.stats["pop_push_ld"] = self.stats.get("pop_push_ld", 0) + 1
                        i += 3
                        continue

                # ld hl,0; ld a,l; ld (addr),a -> xor a; ld (addr),a; ld hl,0
                if opcode == "ld" and operands.lower() == "hl,0":
                    if i + 2 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        p2 = self._parse_line(lines[i + 2].strip())
                        if (p1 and p1[0] == "ld" and p1[1].lower() == "a,l" and
                            p2 and p2[0] == "ld" and p2[1].startswith("(") and p2[1].lower().endswith("),a")):
                            addr = p2[1][:-2]  # Get (addr) part
                            result.append("\txor a")
                            result.append(f"\tld {addr},a")
                            result.append("\tld hl,0")
                            changed = True
                            self.stats["xor_a_store"] = self.stats.get("xor_a_store", 0) + 1
                            i += 3
                            continue

                # ld hl,(addr1); push hl; ld hl,(addr2); ex de,hl; pop hl
                # -> ld de,(addr2); ld hl,(addr1)
                if opcode == "ld" and operands.lower().startswith("hl,(") and operands.endswith(")"):
                    addr1 = operands[3:]  # Keep the (addr) part
                    if i + 4 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        p2 = self._parse_line(lines[i + 2].strip())
                        p3 = self._parse_line(lines[i + 3].strip())
                        p4 = self._parse_line(lines[i + 4].strip())
                        if (p1 and p1[0] == "push" and p1[1].lower() == "hl" and
                            p2 and p2[0] == "ld" and p2[1].lower().startswith("hl,(") and
                            p3 and p3[0] == "ex" and p3[1].lower() == "de,hl" and
                            p4 and p4[0] == "pop" and p4[1].lower() == "hl"):
                            addr2 = p2[1][3:]  # Get (addr2)
                            result.append(f"\tld de,{addr2}")
                            result.append(f"\tld hl,{addr1}")
                            changed = True
                            self.stats["ld_de_nn"] = self.stats.get("ld_de_nn", 0) + 1
                            i += 5
                            continue

            result.append(lines[i])
            i += 1

        return result, changed

    def _convert_to_relative_jumps(self, lines: list[str]) -> list[str]:
        """Convert jp to jr where the jump is within range."""
        # First pass: find all label positions
        label_lines: dict[str, int] = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("\t"):
                label = stripped.split(":")[0].strip()
                label_lines[label] = i

        # Second pass: convert jumps where target is close
        result: list[str] = []
        for i, line in enumerate(lines):
            parsed = self._parse_line(line.strip())

            if parsed:
                opcode, operands = parsed

                # Check for convertible jumps
                convert_map = {
                    "jp": ("jr", None),
                    "jp z,": ("jr z,", 5),
                    "jp nz,": ("jr nz,", 6),
                    "jp c,": ("jr c,", 5),
                    "jp nc,": ("jr nc,", 6),
                }

                for jp_prefix, (jr_prefix, prefix_len) in convert_map.items():
                    if prefix_len:
                        if opcode == "jp" and operands.lower().startswith(jp_prefix[3:]):
                            # Conditional jump
                            target = operands[prefix_len - 3:].strip()
                            if target in label_lines:
                                distance = label_lines[target] - i
                                # Conservative estimate: ~40 lines is roughly 125 bytes
                                if -40 < distance < 40:
                                    result.append(f"\t{jr_prefix}{target}")
                                    self.stats["jr_convert"] = self.stats.get("jr_convert", 0) + 1
                                    break
                    else:
                        if opcode == "jp" and "," not in operands and operands.lower() != "(hl)":
                            # Unconditional jp to label
                            target = operands.strip()
                            if target in label_lines:
                                distance = label_lines[target] - i
                                if -40 < distance < 40:
                                    result.append(f"\tjr {target}")
                                    self.stats["jr_convert"] = self.stats.get("jr_convert", 0) + 1
                                    break
                else:
                    result.append(line)
                    continue
                continue

            result.append(line)

        return result

    def _jump_threading_pass(self, lines: list[str]) -> tuple[list[str], bool]:
        """
        Jump threading optimization.

        If a jump targets a label whose only content is another unconditional jump,
        thread through to the final destination.
        """
        changed = False

        # Build map of label -> (line index, first instruction after label)
        label_info: dict[str, tuple[int, str | None]] = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("\t"):
                label = stripped.split(":")[0].strip()
                # Find first instruction after this label
                first_instr = None
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith(";"):
                        continue
                    if ":" in next_line and not next_line.startswith("\t"):
                        break
                    first_instr = next_line
                    break
                label_info[label] = (i, first_instr)

        # Build map of label -> final destination
        label_target: dict[str, str] = {}
        for label, (_, first_instr) in label_info.items():
            if first_instr:
                parsed = self._parse_line(first_instr)
                if parsed and parsed[0] in ("jp", "jr") and "," not in parsed[1] and parsed[1].lower() != "(hl)":
                    target = parsed[1].strip()
                    # Follow the chain
                    visited = {label}
                    while target in label_info and target not in visited:
                        visited.add(target)
                        _, target_instr = label_info[target]
                        if target_instr:
                            target_parsed = self._parse_line(target_instr)
                            if target_parsed and target_parsed[0] in ("jp", "jr") and "," not in target_parsed[1] and target_parsed[1].lower() != "(hl)":
                                target = target_parsed[1].strip()
                            else:
                                break
                        else:
                            break
                    if target != label:
                        label_target[label] = target

        # Track which labels are referenced
        label_refs: dict[str, int] = {label: 0 for label in label_info}

        # Rewrite jumps to use final destinations
        result: list[str] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            parsed = self._parse_line(stripped)

            if parsed and parsed[0] in ("jp", "jr", "call", "djnz"):
                operands = parsed[1]
                # Handle conditional jumps
                if "," in operands:
                    parts = operands.split(",", 1)
                    target = parts[1].strip()
                    prefix = parts[0] + ","
                else:
                    target = operands.strip()
                    prefix = ""

                # Thread through for unconditional jumps only
                if parsed[0] in ("jp", "jr") and not prefix and target in label_target:
                    new_target = label_target[target]
                    if parsed[0] == "jp":
                        result.append(f"\tjp {new_target}")
                    else:
                        result.append(f"\tjr {new_target}")
                    changed = True
                    self.stats["jump_thread"] = self.stats.get("jump_thread", 0) + 1
                    label_refs[new_target] = label_refs.get(new_target, 0) + 1
                else:
                    result.append(line)
                    if target in label_refs:
                        label_refs[target] += 1
            elif parsed and parsed[0] == "dw":
                # Thread dw references
                target = parsed[1].strip()
                if target in label_target:
                    new_target = label_target[target]
                    result.append(f"\tdw\t{new_target}")
                    changed = True
                    self.stats["dw_thread"] = self.stats.get("dw_thread", 0) + 1
                    label_refs[new_target] = label_refs.get(new_target, 0) + 1
                else:
                    result.append(line)
                    if target in label_refs:
                        label_refs[target] += 1
            else:
                result.append(line)
                if ":" in stripped and not stripped.startswith("\t"):
                    pass
                else:
                    for label in label_info:
                        if label in stripped:
                            label_refs[label] = label_refs.get(label, 0) + 1

        # Remove unreferenced labels that just jump
        final_result: list[str] = []
        i = 0
        while i < len(result):
            line = result[i]
            stripped = line.strip()

            if ":" in stripped and not stripped.startswith("\t"):
                label = stripped.split(":")[0].strip()

                if label in label_refs and label_refs[label] == 0 and label in label_target:
                    # Check if previous instruction prevents fall-through
                    can_fallthrough = True
                    for j in range(len(final_result) - 1, -1, -1):
                        prev = final_result[j].strip()
                        if not prev or prev.startswith(";"):
                            continue
                        if ":" in prev and not prev.startswith("\t"):
                            break
                        prev_parsed = self._parse_line(prev)
                        if prev_parsed:
                            if prev_parsed[0] in ("jp", "jr", "ret") and "," not in prev_parsed[1]:
                                can_fallthrough = False
                            break

                    if not can_fallthrough:
                        changed = True
                        self.stats["dead_label_removed"] = self.stats.get("dead_label_removed", 0) + 1
                        i += 1
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

    def _dead_store_elimination(self, lines: list[str]) -> tuple[list[str], bool]:
        """
        Eliminate dead stores at procedure entry.

        Pattern: A procedure stores a register parameter to memory at entry,
        but uses the register directly without ever loading from that memory.
        """
        result: list[str] = []
        changed = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Look for procedure entry (label followed by ld (addr),a)
            if ":" in stripped and not stripped.startswith("\t") and not stripped.startswith(";"):
                label = stripped.split(":")[0].strip()
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    parsed = self._parse_line(next_stripped)
                    # Check for ld (addr),a pattern
                    if (parsed and parsed[0] == "ld" and
                        parsed[1].startswith("(") and parsed[1].lower().endswith("),a")):
                        addr = parsed[1][1:-3]  # Extract addr from (addr),a
                        # Find end of procedure - look for next top-level procedure
                        # (non-nested label that doesn't start with @ or ?? and isn't
                        # preceded by whitespace)
                        proc_end = i + 2
                        proc_label = label
                        while proc_end < len(lines):
                            end_stripped = lines[proc_end].strip()
                            if (":" in end_stripped and
                                not end_stripped.startswith("\t") and
                                not end_stripped.startswith(";")):
                                lbl = end_stripped.split(":")[0].strip()
                                # Stop at next top-level procedure (not starting with @ or ??)
                                # Nested procedures start with @ and internal labels with ??
                                if not lbl.startswith("@") and not lbl.startswith("??"):
                                    break
                            proc_end += 1

                        # Check if addr is ever loaded within this procedure
                        # (including nested procedures that may access enclosing params)
                        addr_loaded = False
                        for j in range(i + 2, proc_end):
                            check_line = lines[j].strip()
                            if f"({addr})" in check_line.lower() or f"({addr.lower()})" in check_line.lower():
                                p = self._parse_line(check_line)
                                if p and p[0] == "ld":
                                    if not p[1].startswith("("):
                                        addr_loaded = True
                                        break

                        if not addr_loaded:
                            result.append(line)  # Keep the label
                            i += 2  # Skip the store instruction
                            changed = True
                            self.stats["dead_store_elim"] = self.stats.get("dead_store_elim", 0) + 1
                            continue

            result.append(line)
            i += 1

        return result, changed

    def _parse_line(self, line: str) -> tuple[str, str] | None:
        """Parse a Z80 assembly line into (opcode, operands)."""
        line = line.strip()

        if not line or line.startswith(";"):
            return None

        # Handle labels with potential instruction after
        if ":" in line and not line.startswith("\t"):
            parts = line.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                line = parts[1].strip()
            else:
                return None

        # Skip directives (but not dw which we may need to thread)
        directives = {"org", "end", "db", "ds", "equ", "public", "extrn"}

        parts = line.split(None, 1)
        if not parts:
            return None

        opcode = parts[0].lower()
        if opcode in directives:
            return None

        operands = parts[1].split(";")[0].strip() if len(parts) > 1 else ""

        return (opcode, operands)

    def _parse_const(self, s: str) -> int | None:
        """Parse an assembly constant value. Returns None if not a constant."""
        s = s.strip().upper()
        if not s:
            return None
        # Skip if it's a label/symbol reference (starts with letter but not a number format)
        if s[0].isalpha() and not s.endswith('H') and not s.endswith('B') and not s.endswith('O'):
            return None
        try:
            # Handle hex (0FFH, 10H, 0x10)
            if s.endswith('H'):
                return int(s[:-1], 16)
            # Handle binary (10101B)
            if s.endswith('B') and all(c in '01' for c in s[:-1]):
                return int(s[:-1], 2)
            # Handle octal (77O, 77Q)
            if s.endswith('O') or s.endswith('Q'):
                return int(s[:-1], 8)
            # Handle 0x prefix
            if s.startswith('0X'):
                return int(s, 16)
            # Plain decimal
            return int(s)
        except ValueError:
            return None

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
                    pat_re = pat_operands.replace("*", ".*")
                    if not re.match(pat_re, inst_operands, re.IGNORECASE):
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
