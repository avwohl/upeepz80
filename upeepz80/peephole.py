"""
Peephole Optimizer for Z80.

Performs pattern-based optimizations on generated Z80 assembly code.
This runs after code generation to clean up inefficient sequences.

This module expects pure Z80 mnemonics as input (LD, JP, JR, etc.)
and produces optimized Z80 assembly as output.

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

    This optimizer expects pure Z80 mnemonics (LD, JP, JR, etc.)
    as input and produces Z80 assembly output.
    """

    def __init__(self) -> None:
        self.patterns = self._init_patterns()
        self.stats: dict[str, int] = {}

    def _init_patterns(self) -> list[PeepholePattern]:
        """Initialize Z80 peephole optimization patterns."""
        return [
            # Push/Pop elimination: PUSH rr; POP rr -> (nothing)
            PeepholePattern(
                name="push_pop_same",
                pattern=[("PUSH", None), ("POP", None)],
                replacement=[],
                condition=lambda ops: ops[0][1].upper() == ops[1][1].upper(),
            ),
            # Redundant LD: LD A,r; LD r,A -> LD A,r
            PeepholePattern(
                name="redundant_ld",
                pattern=[("LD", "A,*"), ("LD", "*,A")],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1].split(",")[1].upper() == ops[1][1].split(",")[0].upper(),
            ),
            # Zero A: LD A,0 -> XOR A (smaller, faster)
            PeepholePattern(
                name="zero_a_ld",
                pattern=[("LD", "A,0")],
                replacement=[("XOR", "A")],
            ),
            # Compare to zero: CP 0 -> OR A (sets Z flag, smaller)
            PeepholePattern(
                name="cp_zero",
                pattern=[("CP", "0")],
                replacement=[("OR", "A")],
            ),
            # Redundant duplicate LD: LD X,Y; LD X,Y -> LD X,Y
            PeepholePattern(
                name="duplicate_ld",
                pattern=[("LD", None), ("LD", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: ops[0][1].upper() == ops[1][1].upper(),
            ),
            # LD A,A -> (nothing, useless)
            PeepholePattern(
                name="ld_a_a",
                pattern=[("LD", "A,A")],
                replacement=[],
            ),
            # LD B,B, LD C,C, etc. -> (nothing)
            PeepholePattern(
                name="ld_r_r",
                pattern=[("LD", None)],
                replacement=[],
                condition=lambda ops: len(ops[0][1].split(",")) == 2 and
                                      ops[0][1].split(",")[0].strip().upper() ==
                                      ops[0][1].split(",")[1].strip().upper() and
                                      ops[0][1].split(",")[0].strip().upper() in
                                      ("A", "B", "C", "D", "E", "H", "L"),
            ),
            # INC A; DEC A -> (nothing)
            PeepholePattern(
                name="inc_dec_a",
                pattern=[("INC", "A"), ("DEC", "A")],
                replacement=[],
            ),
            # DEC A; INC A -> (nothing)
            PeepholePattern(
                name="dec_inc_a",
                pattern=[("DEC", "A"), ("INC", "A")],
                replacement=[],
            ),
            # INC HL; DEC HL -> (nothing)
            PeepholePattern(
                name="inc_dec_hl",
                pattern=[("INC", "HL"), ("DEC", "HL")],
                replacement=[],
            ),
            # DEC HL; INC HL -> (nothing)
            PeepholePattern(
                name="dec_inc_hl",
                pattern=[("DEC", "HL"), ("INC", "HL")],
                replacement=[],
            ),
            # INC DE; DEC DE -> (nothing)
            PeepholePattern(
                name="inc_dec_de",
                pattern=[("INC", "DE"), ("DEC", "DE")],
                replacement=[],
            ),
            # DEC DE; INC DE -> (nothing)
            PeepholePattern(
                name="dec_inc_de",
                pattern=[("DEC", "DE"), ("INC", "DE")],
                replacement=[],
            ),
            # INC BC; DEC BC -> (nothing)
            PeepholePattern(
                name="inc_dec_bc",
                pattern=[("INC", "BC"), ("DEC", "BC")],
                replacement=[],
            ),
            # DEC BC; INC BC -> (nothing)
            PeepholePattern(
                name="dec_inc_bc",
                pattern=[("DEC", "BC"), ("INC", "BC")],
                replacement=[],
            ),
            # OR A; OR A -> OR A
            PeepholePattern(
                name="double_or_a",
                pattern=[("OR", "A"), ("OR", "A")],
                replacement=[("OR", "A")],
            ),
            # AND A; AND A -> AND A
            PeepholePattern(
                name="double_and_a",
                pattern=[("AND", "A"), ("AND", "A")],
                replacement=[("AND", "A")],
            ),
            # XOR A; XOR A -> XOR A (still zero)
            PeepholePattern(
                name="double_xor_a",
                pattern=[("XOR", "A"), ("XOR", "A")],
                replacement=[("XOR", "A")],
            ),
            # EX DE,HL; EX DE,HL -> (nothing)
            PeepholePattern(
                name="double_ex",
                pattern=[("EX", "DE,HL"), ("EX", "DE,HL")],
                replacement=[],
            ),
            # EX (SP),HL; EX (SP),HL -> (nothing)
            PeepholePattern(
                name="double_ex_sp",
                pattern=[("EX", "(SP),HL"), ("EX", "(SP),HL")],
                replacement=[],
            ),
            # CCF; CCF -> (nothing) - complement carry twice
            PeepholePattern(
                name="double_ccf",
                pattern=[("CCF", ""), ("CCF", "")],
                replacement=[],
            ),
            # CPL; CPL -> (nothing) - complement A twice
            PeepholePattern(
                name="double_cpl",
                pattern=[("CPL", ""), ("CPL", "")],
                replacement=[],
            ),
            # PUSH HL; POP DE -> LD D,H; LD E,L (faster: 21 cycles -> 8 cycles)
            PeepholePattern(
                name="push_pop_copy_hl_de",
                pattern=[("PUSH", "HL"), ("POP", "DE")],
                replacement=[("LD", "D,H"), ("LD", "E,L")],
            ),
            # PUSH DE; POP HL -> LD H,D; LD L,E
            PeepholePattern(
                name="push_pop_copy_de_hl",
                pattern=[("PUSH", "DE"), ("POP", "HL")],
                replacement=[("LD", "H,D"), ("LD", "L,E")],
            ),
            # PUSH BC; POP DE -> LD D,B; LD E,C
            PeepholePattern(
                name="push_pop_copy_bc_de",
                pattern=[("PUSH", "BC"), ("POP", "DE")],
                replacement=[("LD", "D,B"), ("LD", "E,C")],
            ),
            # PUSH BC; POP HL -> LD H,B; LD L,C
            PeepholePattern(
                name="push_pop_copy_bc_hl",
                pattern=[("PUSH", "BC"), ("POP", "HL")],
                replacement=[("LD", "H,B"), ("LD", "L,C")],
            ),
            # PUSH HL; POP BC -> LD B,H; LD C,L
            PeepholePattern(
                name="push_pop_copy_hl_bc",
                pattern=[("PUSH", "HL"), ("POP", "BC")],
                replacement=[("LD", "B,H"), ("LD", "C,L")],
            ),
            # PUSH DE; POP BC -> LD B,D; LD C,E
            PeepholePattern(
                name="push_pop_copy_de_bc",
                pattern=[("PUSH", "DE"), ("POP", "BC")],
                replacement=[("LD", "B,D"), ("LD", "C,E")],
            ),
            # CCF; SCF -> SCF (set carry directly)
            PeepholePattern(
                name="ccf_scf",
                pattern=[("CCF", None), ("SCF", None)],
                replacement=[("SCF", "")],
            ),
            # CALL x; RET -> JP x (tail call optimization)
            PeepholePattern(
                name="tail_call",
                pattern=[("CALL", None), ("RET", "")],
                replacement=None,  # Replaced specially
                condition=lambda ops: True,
            ),
            # RET; RET -> RET (unreachable code)
            PeepholePattern(
                name="double_ret",
                pattern=[("RET", ""), ("RET", "")],
                replacement=[("RET", "")],
            ),
            # LD A,(HL); LD E,A -> LD E,(HL)
            PeepholePattern(
                name="ld_a_hl_ld_ea",
                pattern=[("LD", "A,(HL)"), ("LD", "E,A")],
                replacement=[("LD", "E,(HL)")],
            ),
            # LD A,(HL); LD D,A -> LD D,(HL)
            PeepholePattern(
                name="ld_a_hl_ld_da",
                pattern=[("LD", "A,(HL)"), ("LD", "D,A")],
                replacement=[("LD", "D,(HL)")],
            ),
            # LD A,(HL); LD C,A -> LD C,(HL)
            PeepholePattern(
                name="ld_a_hl_ld_ca",
                pattern=[("LD", "A,(HL)"), ("LD", "C,A")],
                replacement=[("LD", "C,(HL)")],
            ),
            # LD A,(HL); LD B,A -> LD B,(HL)
            PeepholePattern(
                name="ld_a_hl_ld_ba",
                pattern=[("LD", "A,(HL)"), ("LD", "B,A")],
                replacement=[("LD", "B,(HL)")],
            ),
            # LD B,A; LD A,B -> LD B,A
            PeepholePattern(
                name="ld_ba_ab",
                pattern=[("LD", "B,A"), ("LD", "A,B")],
                replacement=[("LD", "B,A")],
            ),
            # LD C,A; LD A,C -> LD C,A
            PeepholePattern(
                name="ld_ca_ac",
                pattern=[("LD", "C,A"), ("LD", "A,C")],
                replacement=[("LD", "C,A")],
            ),
            # LD D,A; LD A,D -> LD D,A
            PeepholePattern(
                name="ld_da_ad",
                pattern=[("LD", "D,A"), ("LD", "A,D")],
                replacement=[("LD", "D,A")],
            ),
            # LD E,A; LD A,E -> LD E,A
            PeepholePattern(
                name="ld_ea_ae",
                pattern=[("LD", "E,A"), ("LD", "A,E")],
                replacement=[("LD", "E,A")],
            ),
            # LD H,A; LD A,H -> LD H,A
            PeepholePattern(
                name="ld_ha_ah",
                pattern=[("LD", "H,A"), ("LD", "A,H")],
                replacement=[("LD", "H,A")],
            ),
            # LD L,A; LD A,L -> LD L,A
            PeepholePattern(
                name="ld_la_al",
                pattern=[("LD", "L,A"), ("LD", "A,L")],
                replacement=[("LD", "L,A")],
            ),
            # LD (addr),HL; LD HL,(addr) -> LD (addr),HL (same address)
            PeepholePattern(
                name="ld_store_load_same",
                pattern=[("LD", None), ("LD", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: (ops[0][1].startswith("(") and
                                       ops[0][1].endswith("),HL") and
                                       ops[1][1] == f"HL,{ops[0][1][:-3]}"),
            ),
            # LD (addr),A; LD A,(addr) -> LD (addr),A (same address)
            PeepholePattern(
                name="sta_lda_same",
                pattern=[("LD", None), ("LD", None)],
                replacement=None,  # Keep first only
                condition=lambda ops: (ops[0][1].startswith("(") and
                                       ops[0][1].endswith("),A") and
                                       ops[1][1] == f"A,{ops[0][1][:-2]}"),
            ),
            # AND 0FFH -> OR A (same effect, smaller)
            PeepholePattern(
                name="and_ff",
                pattern=[("AND", "0FFH")],
                replacement=[("OR", "A")],
            ),
            # OR 0 -> OR A (same effect)
            PeepholePattern(
                name="or_0",
                pattern=[("OR", "0")],
                replacement=[("OR", "A")],
            ),
            # XOR 0 -> OR A (same effect, sets flags)
            PeepholePattern(
                name="xor_0",
                pattern=[("XOR", "0")],
                replacement=[("OR", "A")],
            ),
            # PUSH HL; EX DE,HL; POP HL -> LD D,H; LD E,L
            # The EX swaps HL<->DE, then POP restores HL, so DE = original HL
            PeepholePattern(
                name="push_ex_pop",
                pattern=[("PUSH", "HL"), ("EX", "DE,HL"), ("POP", "HL")],
                replacement=[("LD", "D,H"), ("LD", "E,L")],
            ),
            # LD H,0; LD D,H; LD E,L -> LD D,0; LD E,L
            # D = H = 0, so just load D directly with 0
            PeepholePattern(
                name="ld_h0_dh_el",
                pattern=[("LD", "H,0"), ("LD", "D,H"), ("LD", "E,L")],
                replacement=[("LD", "D,0"), ("LD", "E,L")],
            ),
            # Wasteful byte extension before byte op: LD L,A; LD H,0; SUB x -> SUB x
            # (Also for CP, AND, OR, XOR, ADD byte ops)
            PeepholePattern(
                name="useless_extend_before_sub",
                pattern=[("LD", "L,A"), ("LD", "H,0"), ("SUB", None)],
                replacement=None,  # Keep last only
                condition=lambda ops: True,  # Always apply
            ),
            PeepholePattern(
                name="useless_extend_before_cp",
                pattern=[("LD", "L,A"), ("LD", "H,0"), ("CP", None)],
                replacement=None,  # Keep last only
                condition=lambda ops: True,
            ),
            # Redundant byte extension: LD L,A; LD H,0; LD L,A; LD H,0 -> LD L,A; LD H,0
            PeepholePattern(
                name="double_byte_extend",
                pattern=[("LD", "L,A"), ("LD", "H,0"), ("LD", "L,A"), ("LD", "H,0")],
                replacement=[("LD", "L,A"), ("LD", "H,0")],
            ),
            # Redundant load after push: LD L,A; LD H,0; PUSH HL; LD L,A -> LD L,A; LD H,0; PUSH HL
            PeepholePattern(
                name="redundant_ld_l_after_push",
                pattern=[("LD", "L,A"), ("LD", "H,0"), ("PUSH", "HL"), ("LD", "L,A")],
                replacement=[("LD", "L,A"), ("LD", "H,0"), ("PUSH", "HL")],
            ),
            # LD HL,0FFFFH; LD A,L; OR H -> LD HL,0FFFFH; OR A
            # Since 0xFFFF is always true
            PeepholePattern(
                name="test_true_const",
                pattern=[("LD", "HL,0FFFFH"), ("LD", "A,L"), ("OR", "H")],
                replacement=[("LD", "HL,0FFFFH"), ("OR", "A")],
            ),
            # LD HL,1; LD A,L; OR H -> LD A,1; OR A (smaller)
            PeepholePattern(
                name="test_true_const_1",
                pattern=[("LD", "HL,1"), ("LD", "A,L"), ("OR", "H")],
                replacement=[("LD", "A,1"), ("OR", "A")],
            ),
            # LD HL,1; LD C,L -> LD C,1 (for shift count)
            PeepholePattern(
                name="ld_h1_cl",
                pattern=[("LD", "HL,1"), ("LD", "C,L")],
                replacement=[("LD", "C,1")],
            ),
            # LD HL,0; LD A,L; OR H -> XOR A (sets Z, clears A)
            PeepholePattern(
                name="test_false_const",
                pattern=[("LD", "HL,0"), ("LD", "A,L"), ("OR", "H")],
                replacement=[("XOR", "A")],
            ),
            # PUSH HL; LD (addr),HL; POP HL -> LD (addr),HL
            # LD (addr),HL doesn't modify HL
            PeepholePattern(
                name="push_shld_pop",
                pattern=[("PUSH", "HL"), ("LD", None), ("POP", "HL")],
                replacement=None,  # Keep middle only
                condition=lambda ops: ops[1][1].startswith("(") and ops[1][1].endswith("),HL"),
            ),
            # PUSH AF; LD (addr),A; POP AF -> LD (addr),A
            # Saving/restoring A around a store of A is pointless
            PeepholePattern(
                name="push_sta_pop",
                pattern=[("PUSH", "AF"), ("LD", None), ("POP", "AF")],
                replacement=None,  # Keep middle only
                condition=lambda ops: ops[1][1].startswith("(") and ops[1][1].endswith("),A"),
            ),
            # LD A,L; LD H,0; LD (addr),A -> LD A,L; LD (addr),A
            # MVI H,0 is useless before store
            PeepholePattern(
                name="ld_al_h0_sta",
                pattern=[("LD", "A,L"), ("LD", "H,0"), ("LD", None)],
                replacement=None,  # Keep LD A,L and LD (addr),A
                condition=lambda ops: ops[2][1].startswith("(") and ops[2][1].endswith("),A"),
            ),
            # LD L,A; LD H,0; LD (addr),A -> LD (addr),A
            # If we're just storing A, no need to extend to HL first
            PeepholePattern(
                name="ld_la_h0_sta",
                pattern=[("LD", "L,A"), ("LD", "H,0"), ("LD", None)],
                replacement=None,  # Keep only store
                condition=lambda ops: ops[2][1].startswith("(") and ops[2][1].endswith("),A"),
            ),
            # LD A,L; LD H,0; OR H -> LD A,L; OR A
            # H is 0, so OR H is same as OR A but OR A is smaller
            PeepholePattern(
                name="ld_al_h0_or_h",
                pattern=[("LD", "A,L"), ("LD", "H,0"), ("OR", "H")],
                replacement=[("LD", "A,L"), ("OR", "A")],
            ),
            # LD H,0; OR H -> LD H,0; OR A
            PeepholePattern(
                name="ld_h0_or_h",
                pattern=[("LD", "H,0"), ("OR", "H")],
                replacement=[("LD", "H,0"), ("OR", "A")],
            ),
            # Conditional jump followed by unconditional to same place
            # JP Z,L; JP L -> JP L
            PeepholePattern(
                name="cond_uncond_same_z",
                pattern=[("JP", None), ("JP", None)],
                replacement=None,  # Keep second only
                condition=lambda ops: ops[0][1].startswith("Z,") and ops[0][1][2:] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_nz",
                pattern=[("JP", None), ("JP", None)],
                replacement=None,
                condition=lambda ops: ops[0][1].startswith("NZ,") and ops[0][1][3:] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_c",
                pattern=[("JP", None), ("JP", None)],
                replacement=None,
                condition=lambda ops: ops[0][1].startswith("C,") and ops[0][1][2:] == ops[1][1],
            ),
            PeepholePattern(
                name="cond_uncond_same_nc",
                pattern=[("JP", None), ("JP", None)],
                replacement=None,
                condition=lambda ops: ops[0][1].startswith("NC,") and ops[0][1][3:] == ops[1][1],
            ),
            # LD A,(addr); CP y; JP Z,z; LD A,(addr) -> LD A,(addr); CP y; JP Z,z
            # A unchanged after CP/Jcond
            PeepholePattern(
                name="lda_cp_jz_lda_same",
                pattern=[("LD", None), ("CP", None), ("JP", None), ("LD", None)],
                replacement=None,  # Keep first 3 only
                condition=lambda ops: (ops[0][1].startswith("A,(") and
                                       ops[2][1].startswith("Z,") and
                                       ops[0][1] == ops[3][1]),
            ),
            PeepholePattern(
                name="lda_cp_jnz_lda_same",
                pattern=[("LD", None), ("CP", None), ("JP", None), ("LD", None)],
                replacement=None,
                condition=lambda ops: (ops[0][1].startswith("A,(") and
                                       ops[2][1].startswith("NZ,") and
                                       ops[0][1] == ops[3][1]),
            ),
            # LD A,(addr); OR A; JP Z,z; LD A,(addr) -> LD A,(addr); OR A; JP Z,z
            PeepholePattern(
                name="lda_or_jz_lda_same",
                pattern=[("LD", None), ("OR", "A"), ("JP", None), ("LD", None)],
                replacement=None,
                condition=lambda ops: (ops[0][1].startswith("A,(") and
                                       ops[2][1].startswith("Z,") and
                                       ops[0][1] == ops[3][1]),
            ),
            PeepholePattern(
                name="lda_or_jnz_lda_same",
                pattern=[("LD", None), ("OR", "A"), ("JP", None), ("LD", None)],
                replacement=None,
                condition=lambda ops: (ops[0][1].startswith("A,(") and
                                       ops[2][1].startswith("NZ,") and
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

            if stripped.startswith('.') or stripped.upper().startswith(('ORG', 'EQU', 'DB', 'DW', 'DS')):
                result.append(line)
                i += 1
                continue

            # Special case: JP/JR to immediately following label
            parsed = self._parse_line(lines[i])
            if parsed and parsed[0] in ("JP", "JR") and "," not in parsed[1] and parsed[1] != "(HL)":
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
                        # CALL x; RET -> JP x
                        call_target = instrs[0][1]
                        result.append(f"    JP {call_target}")
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

                # LD A,0 -> XOR A (1 byte vs 2)
                if opcode == "LD" and operands == "A,0":
                    result.append("\tXOR A")
                    changed = True
                    self.stats["xor_a"] = self.stats.get("xor_a", 0) + 1
                    i += 1
                    continue

                # LD A,(addr); INC A; LD (addr),A -> LD HL,addr; INC (HL)
                if opcode == "LD" and operands.startswith("A,(") and operands.endswith(")"):
                    addr = operands[3:-1]  # Extract address
                    if i + 2 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        p2 = self._parse_line(lines[i + 2].strip())
                        if (p1 and p1[0] == "INC" and p1[1] == "A" and
                            p2 and p2[0] == "LD" and p2[1] == f"({addr}),A"):
                            result.append(f"\tLD HL,{addr}")
                            result.append("\tINC (HL)")
                            changed = True
                            self.stats["inc_mem"] = self.stats.get("inc_mem", 0) + 1
                            i += 3
                            continue
                        # Also check for DEC A
                        if (p1 and p1[0] == "DEC" and p1[1] == "A" and
                            p2 and p2[0] == "LD" and p2[1] == f"({addr}),A"):
                            result.append(f"\tLD HL,{addr}")
                            result.append("\tDEC (HL)")
                            changed = True
                            self.stats["dec_mem"] = self.stats.get("dec_mem", 0) + 1
                            i += 3
                            continue

                # EX DE,HL; EX DE,HL -> (nothing)
                if opcode == "EX" and operands == "DE,HL" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "EX" and next_parsed[1] == "DE,HL":
                        changed = True
                        self.stats["double_ex"] = self.stats.get("double_ex", 0) + 1
                        i += 2
                        continue

                # INC HL; DEC HL -> (nothing)
                if opcode == "INC" and operands == "HL" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "DEC" and next_parsed[1] == "HL":
                        changed = True
                        self.stats["inc_dec_hl"] = self.stats.get("inc_dec_hl", 0) + 1
                        i += 2
                        continue

                # DEC HL; INC HL -> (nothing)
                if opcode == "DEC" and operands == "HL" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] == "INC" and next_parsed[1] == "HL":
                        changed = True
                        self.stats["dec_inc_hl"] = self.stats.get("dec_inc_hl", 0) + 1
                        i += 2
                        continue

                # LD (addr),HL; LD HL,(addr) -> LD (addr),HL (same address)
                if opcode == "LD" and operands.startswith("(") and operands.endswith("),HL"):
                    addr = operands[1:-4]
                    if i + 1 < len(lines):
                        next_parsed = self._parse_line(lines[i + 1].strip())
                        if next_parsed and next_parsed[0] == "LD" and next_parsed[1] == f"HL,({addr})":
                            result.append(lines[i])
                            changed = True
                            self.stats["ld_hl_same"] = self.stats.get("ld_hl_same", 0) + 1
                            i += 2
                            continue

                # DEC B; JR/JP NZ,label -> DJNZ label
                if opcode == "DEC" and operands == "B" and i + 1 < len(lines):
                    next_parsed = self._parse_line(lines[i + 1].strip())
                    if next_parsed and next_parsed[0] in ("JR", "JP") and next_parsed[1].startswith("NZ,"):
                        target = next_parsed[1][3:]  # Remove "NZ,"
                        if target in label_lines:
                            distance = label_lines[target] - i
                            if -50 < distance < 50:
                                result.append(f"\tDJNZ {target}")
                                changed = True
                                self.stats["djnz"] = self.stats.get("djnz", 0) + 1
                                i += 2
                                continue

                # PUSH HL; LD HL,(addr); EX DE,HL; POP HL -> LD DE,(addr)
                # Z80 has direct LD DE,(addr) which 8080 doesn't have
                if opcode == "PUSH" and operands == "HL" and i + 3 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    p2 = self._parse_line(lines[i + 2].strip())
                    p3 = self._parse_line(lines[i + 3].strip())
                    if (p1 and p1[0] == "LD" and p1[1].startswith("HL,(") and p1[1].endswith(")") and
                        p2 and p2[0] == "EX" and p2[1] == "DE,HL" and
                        p3 and p3[0] == "POP" and p3[1] == "HL"):
                        addr = p1[1][3:]  # Get (addr) including parens
                        result.append(f"\tLD DE,{addr}")
                        changed = True
                        self.stats["ld_de_addr"] = self.stats.get("ld_de_addr", 0) + 1
                        i += 4
                        continue

                # PUSH AF; LD (addr),A; POP AF -> LD (addr),A
                if opcode == "PUSH" and operands == "AF" and i + 2 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    p2 = self._parse_line(lines[i + 2].strip())
                    if (p1 and p1[0] == "LD" and p1[1].startswith("(") and p1[1].endswith("),A") and
                        p2 and p2[0] == "POP" and p2[1] == "AF"):
                        result.append(lines[i + 1])  # Keep only LD (addr),A
                        changed = True
                        self.stats["push_sta_pop"] = self.stats.get("push_sta_pop", 0) + 1
                        i += 3
                        continue

                # LD HL,const; LD r,L -> LD r,const
                if opcode == "LD" and operands.startswith("HL,") and not operands.startswith("HL,("):
                    const_val = operands[3:]
                    if i + 1 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        if p1 and p1[0] == "LD" and p1[1].endswith(",L"):
                            dest_reg = p1[1][:-2]  # Get destination register
                            if dest_reg in ("A", "B", "C", "D", "E"):
                                result.append(f"\tLD {dest_reg},{const_val}")
                                changed = True
                                self.stats["ld_via_hl"] = self.stats.get("ld_via_hl", 0) + 1
                                i += 2
                                continue

                # POP HL; PUSH HL; LD HL,x -> LD HL,x
                if opcode == "POP" and operands == "HL" and i + 2 < len(lines):
                    p1 = self._parse_line(lines[i + 1].strip())
                    p2 = self._parse_line(lines[i + 2].strip())
                    if (p1 and p1[0] == "PUSH" and p1[1] == "HL" and
                        p2 and p2[0] == "LD" and p2[1].startswith("HL,")):
                        result.append(lines[i + 2])  # Keep only LD HL,x
                        changed = True
                        self.stats["pop_push_ld"] = self.stats.get("pop_push_ld", 0) + 1
                        i += 3
                        continue

                # LD HL,0; LD A,L; LD (addr),A -> XOR A; LD (addr),A; LD HL,0
                if opcode == "LD" and operands == "HL,0":
                    if i + 2 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        p2 = self._parse_line(lines[i + 2].strip())
                        if (p1 and p1[0] == "LD" and p1[1] == "A,L" and
                            p2 and p2[0] == "LD" and p2[1].startswith("(") and p2[1].endswith("),A")):
                            addr = p2[1][:-2]  # Get (addr) part
                            result.append("\tXOR A")
                            result.append(f"\tLD {addr},A")
                            result.append("\tLD HL,0")
                            changed = True
                            self.stats["xor_a_store"] = self.stats.get("xor_a_store", 0) + 1
                            i += 3
                            continue

                # LD HL,(addr1); PUSH HL; LD HL,(addr2); EX DE,HL; POP HL
                # -> LD DE,(addr2); LD HL,(addr1)
                if opcode == "LD" and operands.startswith("HL,(") and operands.endswith(")"):
                    addr1 = operands[3:]  # Keep the (addr) part
                    if i + 4 < len(lines):
                        p1 = self._parse_line(lines[i + 1].strip())
                        p2 = self._parse_line(lines[i + 2].strip())
                        p3 = self._parse_line(lines[i + 3].strip())
                        p4 = self._parse_line(lines[i + 4].strip())
                        if (p1 and p1[0] == "PUSH" and p1[1] == "HL" and
                            p2 and p2[0] == "LD" and p2[1].startswith("HL,(") and
                            p3 and p3[0] == "EX" and p3[1] == "DE,HL" and
                            p4 and p4[0] == "POP" and p4[1] == "HL"):
                            addr2 = p2[1][3:]  # Get (addr2)
                            result.append(f"\tLD DE,{addr2}")
                            result.append(f"\tLD HL,{addr1}")
                            changed = True
                            self.stats["ld_de_nn"] = self.stats.get("ld_de_nn", 0) + 1
                            i += 5
                            continue

            result.append(lines[i])
            i += 1

        return result, changed

    def _convert_to_relative_jumps(self, lines: list[str]) -> list[str]:
        """Convert JP to JR where the jump is within range."""
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
                    "JP": ("JR", None),
                    "JP Z,": ("JR Z,", 5),
                    "JP NZ,": ("JR NZ,", 6),
                    "JP C,": ("JR C,", 5),
                    "JP NC,": ("JR NC,", 6),
                }

                for jp_prefix, (jr_prefix, prefix_len) in convert_map.items():
                    if prefix_len:
                        if opcode == "JP" and operands.startswith(jp_prefix[3:]):
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
                        if opcode == "JP" and "," not in operands and operands != "(HL)":
                            # Unconditional JP to label
                            target = operands.strip()
                            if target in label_lines:
                                distance = label_lines[target] - i
                                if -40 < distance < 40:
                                    result.append(f"\tJR {target}")
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
                if parsed and parsed[0] in ("JP", "JR") and "," not in parsed[1] and parsed[1] != "(HL)":
                    target = parsed[1].strip()
                    # Follow the chain
                    visited = {label}
                    while target in label_info and target not in visited:
                        visited.add(target)
                        _, target_instr = label_info[target]
                        if target_instr:
                            target_parsed = self._parse_line(target_instr)
                            if target_parsed and target_parsed[0] in ("JP", "JR") and "," not in target_parsed[1] and target_parsed[1] != "(HL)":
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

            if parsed and parsed[0] in ("JP", "JR", "CALL", "DJNZ"):
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
                if parsed[0] in ("JP", "JR") and not prefix and target in label_target:
                    new_target = label_target[target]
                    if parsed[0] == "JP":
                        result.append(f"\tJP {new_target}")
                    else:
                        result.append(f"\tJR {new_target}")
                    changed = True
                    self.stats["jump_thread"] = self.stats.get("jump_thread", 0) + 1
                    label_refs[new_target] = label_refs.get(new_target, 0) + 1
                else:
                    result.append(line)
                    if target in label_refs:
                        label_refs[target] += 1
            elif parsed and parsed[0] == "DW":
                # Thread DW references
                target = parsed[1].strip()
                if target in label_target:
                    new_target = label_target[target]
                    result.append(f"\tDW\t{new_target}")
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
                            if prev_parsed[0] in ("JP", "JR", "RET") and "," not in prev_parsed[1]:
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
                            if next_parsed and next_parsed[0] in ("JP", "JR"):
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

            # Look for procedure entry (label followed by LD (addr),A)
            if ":" in stripped and not stripped.startswith("\t") and not stripped.startswith(";"):
                label = stripped.split(":")[0].strip()
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    parsed = self._parse_line(next_stripped)
                    # Check for LD (addr),A pattern
                    if (parsed and parsed[0] == "LD" and
                        parsed[1].startswith("(") and parsed[1].endswith("),A")):
                        addr = parsed[1][1:-3]  # Extract addr from (addr),A
                        # Find end of procedure
                        proc_end = i + 2
                        while proc_end < len(lines):
                            end_stripped = lines[proc_end].strip()
                            if (":" in end_stripped and
                                not end_stripped.startswith("\t") and
                                not end_stripped.startswith(";") and
                                end_stripped.split(":")[0].strip().startswith("@")):
                                break
                            proc_end += 1

                        # Check if addr is ever loaded within this procedure
                        addr_loaded = False
                        for j in range(i + 2, proc_end):
                            check_line = lines[j].strip()
                            if f"({addr})" in check_line:
                                p = self._parse_line(check_line)
                                if p and p[0] == "LD":
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

        # Skip directives
        directives = {"ORG", "END", "DB", "DW", "DS", "EQU", "PUBLIC", "EXTRN"}

        parts = line.split(None, 1)
        if not parts:
            return None

        opcode = parts[0].upper()
        if opcode in directives:
            return None

        operands = parts[1].split(";")[0].strip() if len(parts) > 1 else ""

        return (opcode, operands)

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
                elif pat_operands.upper() != inst_operands.upper():
                    return False

        return True


def optimize(asm_text: str) -> str:
    """Optimize Z80 assembly code.

    This is the main entry point for the optimizer.
    Pass Z80 assembly text (using LD, JP, JR, etc. mnemonics)
    and receive optimized Z80 assembly back.
    """
    optimizer = PeepholeOptimizer()
    return optimizer.optimize(asm_text)
