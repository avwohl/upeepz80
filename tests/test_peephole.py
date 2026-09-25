"""Tests for the upeepz80 peephole optimizer."""

import pytest
from upeepz80 import optimize, PeepholeOptimizer
from upeepz80.peephole import PeepholeOptimizer as _PeepholeOptimizer

# Code after a fragment, overwriting what its rewrite changes: the optimizer
# only rewrites where the registers and flags it changes are dead, and the end
# of the text, like a ret, counts as reading everything.
KILL_DE_FLAGS = "\n\tld de,0\n\tcp b"
KILL_ALL_BUT_HL = "\n\tld a,1\n\tld bc,0\n\tld de,0\n\tcp b"


class TestBasicPatterns:
    """Test basic peephole optimization patterns."""

    def test_ld_a_0_to_xor_a(self):
        """ld a,0 -> xor a, where the flags xor a sets are overwritten"""
        result = optimize("    ld a,0\n    cp b")
        assert result.split() == ["xor", "a", "cp", "b"]

    def test_cp_0_to_or_a(self):
        """cp 0 -> or a, where P/V and N (which differ) are not read"""
        result = optimize("    cp 0\n    sbc a,a")
        assert result.split() == ["or", "a", "sbc", "a,a"]

    def test_push_pop_same_eliminated(self):
        """push rr; pop rr -> nothing"""
        result = optimize("    push hl\n    pop hl")
        assert result.strip() == ""

    def test_push_pop_same_bc(self):
        """push bc; pop bc -> nothing"""
        result = optimize("    push bc\n    pop bc")
        assert result.strip() == ""

    def test_push_pop_same_de(self):
        """push de; pop de -> nothing"""
        result = optimize("    push de\n    pop de")
        assert result.strip() == ""

    def test_push_pop_same_af(self):
        """push af; pop af -> nothing"""
        result = optimize("    push af\n    pop af")
        assert result.strip() == ""

    def test_ld_a_a_eliminated(self):
        """ld a,a -> nothing"""
        result = optimize("    ld a,a")
        assert result.strip() == ""

    def test_ld_b_b_eliminated(self):
        """ld b,b -> nothing"""
        result = optimize("    ld b,b")
        assert result.strip() == ""

    def test_ld_c_c_eliminated(self):
        """ld c,c -> nothing"""
        result = optimize("    ld c,c")
        assert result.strip() == ""


class TestIncDecPatterns:
    """Test inc/dec optimization patterns."""

    def test_inc_dec_a_eliminated(self):
        """inc a; dec a -> nothing"""
        result = optimize("    inc a\n    dec a\n    cp b")
        assert result.strip() == "cp b"

    def test_dec_inc_a_eliminated(self):
        """dec a; inc a -> nothing"""
        result = optimize("    dec a\n    inc a\n    cp b")
        assert result.strip() == "cp b"

    def test_inc_dec_hl_eliminated(self):
        """inc hl; dec hl -> nothing"""
        result = optimize("    inc hl\n    dec hl")
        assert result.strip() == ""

    def test_dec_inc_hl_eliminated(self):
        """dec hl; inc hl -> nothing"""
        result = optimize("    dec hl\n    inc hl")
        assert result.strip() == ""

    def test_inc_dec_de_eliminated(self):
        """inc de; dec de -> nothing"""
        result = optimize("    inc de\n    dec de")
        assert result.strip() == ""

    def test_inc_dec_bc_eliminated(self):
        """inc bc; dec bc -> nothing"""
        result = optimize("    inc bc\n    dec bc")
        assert result.strip() == ""


class TestDoubleInstructionPatterns:
    """Test double instruction elimination patterns."""

    def test_double_or_a(self):
        """or a; or a -> or a"""
        result = optimize("    or a\n    or a")
        assert result.strip() == "or a"

    def test_double_and_a(self):
        """and a; and a -> and a"""
        result = optimize("    and a\n    and a")
        assert result.strip() == "and a"

    def test_double_xor_a(self):
        """xor a; xor a -> xor a"""
        result = optimize("    xor a\n    xor a")
        assert result.strip() == "xor a"

    def test_double_ex_de_hl(self):
        """ex de,hl; ex de,hl -> nothing"""
        result = optimize("    ex de,hl\n    ex de,hl")
        assert result.strip() == ""

    def test_double_ex_sp_hl(self):
        """ex (sp),hl; ex (sp),hl -> nothing"""
        result = optimize("    ex (sp),hl\n    ex (sp),hl")
        assert result.strip() == ""

    def test_double_ccf(self):
        """ccf; ccf -> nothing"""
        result = optimize("    ccf\n    ccf\n    cp b")
        assert result.strip() == "cp b"

    def test_double_cpl(self):
        """cpl; cpl -> nothing"""
        result = optimize("    cpl\n    cpl\n    cp b")
        assert result.strip() == "cp b"

    def test_double_ret(self):
        """ret; ret -> ret"""
        result = optimize("    ret\n    ret")
        assert result.strip() == "ret"


class TestPushPopCopyPatterns:
    """Test push/pop to ld conversion patterns."""

    def test_push_hl_pop_de(self):
        """push hl; pop de -> ld d,h; ld e,l"""
        result = optimize("    push hl\n    pop de")
        assert "ld d,h" in result
        assert "ld e,l" in result

    def test_push_de_pop_hl(self):
        """push de; pop hl -> ld h,d; ld l,e"""
        result = optimize("    push de\n    pop hl")
        assert "ld h,d" in result
        assert "ld l,e" in result

    def test_push_bc_pop_de(self):
        """push bc; pop de -> ld d,b; ld e,c"""
        result = optimize("    push bc\n    pop de")
        assert "ld d,b" in result
        assert "ld e,c" in result

    def test_push_bc_pop_hl(self):
        """push bc; pop hl -> ld h,b; ld l,c"""
        result = optimize("    push bc\n    pop hl")
        assert "ld h,b" in result
        assert "ld l,c" in result

    def test_push_hl_pop_bc(self):
        """push hl; pop bc -> ld b,h; ld c,l"""
        result = optimize("    push hl\n    pop bc")
        assert "ld b,h" in result
        assert "ld c,l" in result

    def test_push_de_pop_bc(self):
        """push de; pop bc -> ld b,d; ld c,e"""
        result = optimize("    push de\n    pop bc")
        assert "ld b,d" in result
        assert "ld c,e" in result


class TestTailCallOptimization:
    """Test tail call optimization."""

    def test_call_ret_to_jp(self):
        """call x; ret -> jp x"""
        result = optimize("    call FUNC\n    ret")
        assert result.strip() == "jp FUNC"

    def test_call_ret_with_label(self):
        """call label; ret -> jp label"""
        result = optimize("    call my_function\n    ret")
        assert result.strip() == "jp my_function"


class TestLoadOptimizations:
    """Test load instruction optimizations."""

    def test_ld_a_hl_ld_e_a(self):
        """ld a,(hl); ld e,a -> ld e,(hl)"""
        result = optimize("    ld a,(hl)\n    ld e,a\n    ld a,c")
        assert result.split("\n")[0].strip() == "ld e,(hl)"

    def test_ld_a_hl_ld_d_a(self):
        """ld a,(hl); ld d,a -> ld d,(hl)"""
        result = optimize("    ld a,(hl)\n    ld d,a\n    ld a,c")
        assert result.split("\n")[0].strip() == "ld d,(hl)"

    def test_ld_a_hl_ld_c_a(self):
        """ld a,(hl); ld c,a -> ld c,(hl)"""
        result = optimize("    ld a,(hl)\n    ld c,a\n    ld a,c")
        assert result.split("\n")[0].strip() == "ld c,(hl)"

    def test_ld_a_hl_ld_b_a(self):
        """ld a,(hl); ld b,a -> ld b,(hl)"""
        result = optimize("    ld a,(hl)\n    ld b,a\n    ld a,c")
        assert result.split("\n")[0].strip() == "ld b,(hl)"

    def test_ld_ba_ab_redundant(self):
        """ld b,a; ld a,b -> ld b,a"""
        result = optimize("    ld b,a\n    ld a,b")
        assert result.strip() == "ld b,a"

    def test_ld_ca_ac_redundant(self):
        """ld c,a; ld a,c -> ld c,a"""
        result = optimize("    ld c,a\n    ld a,c")
        assert result.strip() == "ld c,a"

    def test_ld_da_ad_redundant(self):
        """ld d,a; ld a,d -> ld d,a"""
        result = optimize("    ld d,a\n    ld a,d")
        assert result.strip() == "ld d,a"

    def test_ld_ea_ae_redundant(self):
        """ld e,a; ld a,e -> ld e,a"""
        result = optimize("    ld e,a\n    ld a,e")
        assert result.strip() == "ld e,a"

    def test_duplicate_ld_eliminated(self):
        """ld a,b; ld a,b -> ld a,b"""
        result = optimize("    ld a,b\n    ld a,b")
        assert result.strip() == "ld a,b"


class TestBitwiseOptimizations:
    """Test bitwise operation optimizations."""

    def test_and_ff_to_or_a(self):
        """and 0ffh -> or a"""
        result = optimize("    and 0ffh\n    cp b")
        assert result.split("\n")[0].strip() == "or a"

    def test_or_0_to_or_a(self):
        """or 0 -> or a"""
        result = optimize("    or 0")
        assert result.strip() == "or a"

    def test_xor_0_to_or_a(self):
        """xor 0 -> or a"""
        result = optimize("    xor 0")
        assert result.strip() == "or a"


class TestJumpOptimizations:
    """Test jump optimization patterns."""

    def test_jp_to_next_label_eliminated(self):
        """jp LABEL followed by LABEL: -> nothing"""
        asm = "    jp NEXT\nNEXT:\n    ret"
        result = optimize(asm)
        assert "jp NEXT" not in result
        assert "NEXT:" in result
        assert "ret" in result

    def test_ccf_scf_to_scf(self):
        """ccf; scf -> scf"""
        result = optimize("    ccf\n    scf")
        assert result.strip() == "scf"


class TestConstantOptimizations:
    """Test constant-related optimizations."""

    def test_ld_hl_0_test_to_xor_a(self):
        """ld hl,0; ld a,l; or h -> xor a"""
        result = optimize("    ld hl,0\n    ld a,l\n    or h\n    ld hl,5")
        assert result.split("\n")[0].strip() == "xor a"

    def test_ld_hl_1_ld_c_l(self):
        """ld hl,1; ld c,l -> ld c,1"""
        result = optimize("    ld hl,1\n    ld c,l\n    ld hl,5")
        assert result.split("\n")[0].strip() == "ld c,1"


class TestCaseInsensitivity:
    """Test that input is case-insensitive."""

    def test_uppercase_input(self):
        """Uppercase input should still be optimized."""
        result = optimize("    LD A,0\n    CP B")
        assert "xor" in result.lower()

    def test_mixed_case_input(self):
        """Mixed case input should still be optimized."""
        result = optimize("    Ld A,0\n    Cp B")
        assert "xor" in result.lower()

    def test_uppercase_registers(self):
        """Uppercase registers should work."""
        result = optimize("    push HL\n    pop HL")
        assert result.strip() == ""


class TestOutputCase:
    """Test that output is lowercase."""

    def test_output_opcodes_lowercase(self):
        """All output opcodes should be lowercase."""
        result = optimize("    ld a,0\n    push hl\n    pop de")
        for line in result.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith(';') and ':' not in stripped:
                parts = stripped.split()
                if parts:
                    opcode = parts[0]
                    assert opcode == opcode.lower(), f"Opcode not lowercase: {opcode}"

    def test_output_registers_lowercase(self):
        """All output register names should be lowercase."""
        result = optimize("    push hl\n    pop de")
        assert "ld d,h" in result
        assert "ld e,l" in result
        # Check no uppercase
        assert "D,H" not in result
        assert "E,L" not in result


class TestOptimizerStats:
    """Test optimizer statistics tracking."""

    def test_stats_tracked(self):
        """Optimizer should track statistics."""
        opt = PeepholeOptimizer()
        opt.optimize("    ld a,0\n    cp 0\n    cp b")
        assert opt.stats.get("xor_a", 0) > 0 or opt.stats.get("zero_a_ld", 0) > 0

    def test_stats_accumulate(self):
        """Stats should accumulate across optimizations."""
        opt = PeepholeOptimizer()
        opt.optimize("    ld a,0\n    cp b")
        opt.optimize("    ld a,0\n    cp b")
        total = sum(opt.stats.values())
        assert total >= 2


class TestPreservation:
    """Test that non-optimizable code is preserved."""

    def test_labels_preserved(self):
        """Labels should be preserved."""
        result = optimize("MYLABEL:\n    ret")
        assert "MYLABEL:" in result

    def test_comments_preserved(self):
        """Comments should be preserved."""
        result = optimize("    ld a,b  ; this is a comment")
        assert "comment" in result

    def test_standalone_comments_preserved(self):
        """Standalone comments should be preserved."""
        result = optimize("; this is a comment\n    ret")
        assert "; this is a comment" in result

    def test_directives_preserved(self):
        """Directives should be passed through unchanged."""
        result = optimize("    db 0\n    dw 1234h\n    ret")
        assert "db 0" in result
        assert "dw 1234h" in result
        assert "ret" in result

    @pytest.mark.parametrize("line", ["\tld", "\tld ; note", "\tLD", "X:\tld", "\tinc",
                                      "\tset", "\tres", "\tsrl", "\tldir"])
    def test_an_instruction_without_its_operands_passes_through(self, line):
        """No assembler takes a bare `ld', but the optimizer must not raise
        on one: it looked for the target of a store in operands that were
        not there."""
        result = optimize(f"{line}\n\tld a,(PB)\n\tret\nPB:\n\tds 1\n")
        assert line.split(";")[0].strip() in result


class TestComplexSequences:
    """Test more complex optimization sequences."""

    def test_multiple_optimizations(self):
        """Multiple optimizations should apply."""
        asm = """
    ld a,0
    cp b
    push hl
    pop de
    ret
    ret
"""
        result = optimize(asm)
        assert "xor a" in result
        assert "ld d,h" in result
        assert "ld e,l" in result
        # Only one ret should remain
        assert result.count("ret") == 1

    def test_chained_push_pop(self):
        """Chained push/pop should be optimized."""
        asm = """
    push hl
    pop de
    push de
    pop bc
"""
        result = optimize(asm)
        # Should be converted to register moves
        assert "push" not in result
        assert "pop" not in result

    def test_labels_break_patterns(self):
        """Labels should break pattern matching."""
        asm = """
    push hl
MIDDLE:
    pop hl
"""
        result = optimize(asm)
        # Pattern should not match across label
        assert "push hl" in result
        assert "pop hl" in result


class TestParseConst:
    """Test _parse_const for various number formats."""

    @pytest.fixture
    def opt(self):
        return _PeepholeOptimizer()

    def test_decimal(self, opt):
        assert opt._parse_const("42") == 42

    def test_zero(self, opt):
        assert opt._parse_const("0") == 0

    def test_hex_suffix(self, opt):
        assert opt._parse_const("0FFH") == 255

    def test_hex_suffix_lower(self, opt):
        assert opt._parse_const("10h") == 16

    def test_hex_prefix_0x(self, opt):
        assert opt._parse_const("0x10") == 16

    def test_binary(self, opt):
        assert opt._parse_const("10101B") == 21

    def test_octal_o(self, opt):
        assert opt._parse_const("77O") == 63

    def test_octal_q(self, opt):
        assert opt._parse_const("77Q") == 63

    def test_label_returns_none(self, opt):
        assert opt._parse_const("MYLABEL") is None

    def test_empty_returns_none(self, opt):
        assert opt._parse_const("") is None


class TestIncHlConst:
    """Test ld de,N; add hl,de -> inc hl (repeated) optimization."""

    def test_ld_de_1_add_hl_de(self):
        """ld de,1; add hl,de -> inc hl"""
        result = optimize("\tld de,1\n\tadd hl,de" + KILL_DE_FLAGS)
        assert result.split("\n")[0].strip() == "inc hl"

    def test_ld_de_2_add_hl_de(self):
        """ld de,2; add hl,de -> inc hl; inc hl"""
        result = optimize("\tld de,2\n\tadd hl,de" + KILL_DE_FLAGS)
        assert result.strip().count("inc hl") == 2

    def test_ld_de_3_add_hl_de(self):
        """ld de,3; add hl,de -> inc hl; inc hl; inc hl"""
        result = optimize("\tld de,3\n\tadd hl,de" + KILL_DE_FLAGS)
        assert result.strip().count("inc hl") == 3

    def test_ld_de_4_not_optimized(self):
        """ld de,4; add hl,de should NOT be converted to inc hl."""
        result = optimize("\tld de,4\n\tadd hl,de" + KILL_DE_FLAGS)
        assert "add hl,de" in result


class TestMulStrengthReduction:
    """Test multiply by power-of-2 strength reduction."""

    def test_mul_by_2(self):
        """ld de,2; call ??mul16 -> add hl,hl"""
        result = optimize("\tld de,2\n\tcall ??mul16" + KILL_ALL_BUT_HL)
        assert "add hl,hl" in result
        assert result.strip().count("add hl,hl") == 1
        assert "call" not in result

    def test_mul_by_4(self):
        """ld de,4; call ??mul16 -> add hl,hl; add hl,hl"""
        result = optimize("\tld de,4\n\tcall ??mul16" + KILL_ALL_BUT_HL)
        assert result.strip().count("add hl,hl") == 2
        assert "call" not in result

    def test_mul_by_8(self):
        """ld de,8; call ??mul16 -> 3x add hl,hl"""
        result = optimize("\tld de,8\n\tcall ??mul16" + KILL_ALL_BUT_HL)
        assert result.strip().count("add hl,hl") == 3

    def test_mul_by_non_power_of_2(self):
        """ld de,3; call ??mul16 should NOT be strength-reduced."""
        result = optimize("\tld de,3\n\tcall ??mul16" + KILL_ALL_BUT_HL)
        assert "call" in result

    def test_mul_at_mul16(self):
        """ld de,2; call @mul16 -> add hl,hl"""
        result = optimize("\tld de,2\n\tcall @mul16" + KILL_ALL_BUT_HL)
        assert "add hl,hl" in result

    def test_mul_dunder_mul16(self):
        """ld de,2; call __mul16 -> add hl,hl"""
        result = optimize("\tld de,2\n\tcall __mul16" + KILL_ALL_BUT_HL)
        assert "add hl,hl" in result


class TestIncDecMem:
    """Test ld a,(addr); inc/dec a; ld (addr),a -> ld hl,addr; inc/dec (hl)."""

    def test_inc_mem(self):
        """ld a,(COUNT); inc a; ld (COUNT),a -> ld hl,COUNT; inc (hl)"""
        asm = "\tld a,(COUNT)\n\tinc a\n\tld (COUNT),a\n\tld a,1\n\tld hl,0"
        result = optimize(asm)
        assert "inc (hl)" in result
        assert "ld hl,COUNT" in result

    def test_dec_mem(self):
        """ld a,(COUNT); dec a; ld (COUNT),a -> ld hl,COUNT; dec (hl)"""
        asm = "\tld a,(COUNT)\n\tdec a\n\tld (COUNT),a\n\tld a,1\n\tld hl,0"
        result = optimize(asm)
        assert "dec (hl)" in result
        assert "ld hl,COUNT" in result


class TestDjnz:
    """Test dec b; jr/jp nz,label -> djnz label conversion."""

    def test_dec_b_jr_nz(self):
        """dec b; jr nz,LOOP -> djnz LOOP"""
        asm = "LOOP:\n\tnop\n\tdec b\n\tjr nz,LOOP\n\tcp c"
        result = optimize(asm)
        assert "djnz LOOP" in result
        assert "dec b" not in result

    def test_dec_b_jp_nz(self):
        """dec b; jp nz,LOOP -> djnz LOOP (if in range)"""
        asm = "LOOP:\n\tnop\n\tdec b\n\tjp nz,LOOP\n\tcp c"
        result = optimize(asm)
        assert "djnz LOOP" in result


class TestShiftToZ80:
    """Test 8080-style 16-bit right shift to Z80 native."""

    def test_shr_hl(self):
        """or a; ld a,h; rra; ld h,a; ld a,l; rra; ld l,a -> srl h; rr l"""
        asm = "\tor a\n\tld a,h\n\trra\n\tld h,a\n\tld a,l\n\trra\n\tld l,a\n\tld a,1\n\tcp b"
        result = optimize(asm)
        assert "srl h" in result
        assert "rr l" in result
        assert "rra" not in result


class TestSubdeZero:
    """Test ld de,0; call ??subde -> elimination."""

    def test_subde_zero_eliminated(self):
        """ld de,0; call ??subde -> nothing"""
        result = optimize("\tld de,0\n\tcall ??subde" + KILL_ALL_BUT_HL)
        assert result.strip() == KILL_ALL_BUT_HL.strip()

    def test_subde_zero_at_variant(self):
        """ld de,0; call @subde -> nothing"""
        result = optimize("\tld de,0\n\tcall @subde" + KILL_ALL_BUT_HL)
        assert result.strip() == KILL_ALL_BUT_HL.strip()


class TestLdDeAddr:
    """Test push hl; ld hl,(addr); ex de,hl; pop hl -> ld de,(addr)."""

    def test_ld_de_from_mem(self):
        """push hl; ld hl,(DATA); ex de,hl; pop hl -> ld de,(DATA)"""
        asm = "\tpush hl\n\tld hl,(DATA)\n\tex de,hl\n\tpop hl"
        result = optimize(asm)
        assert "ld de,(DATA)" in result
        assert "push" not in result
        assert "pop" not in result
        assert "ex" not in result


class TestJumpThreading:
    """Test jump threading optimization."""

    def test_thread_through_unconditional(self):
        """jp A where A: jp B -> jp B"""
        asm = "\tjp LBL_A\nLBL_A:\n\tjp LBL_B\nLBL_B:\n\tret"
        result = optimize(asm)
        # Should jump directly to LBL_B
        assert "jp LBL_A" not in result or "jr LBL_A" not in result

    def test_chain_threading(self):
        """jp A; A: jp B; B: jp C -> jp C"""
        asm = "\tjp L1\nL1:\n\tjp L2\nL2:\n\tjp L3\nL3:\n\tret"
        result = optimize(asm)
        # Should not contain jp L1 or jp L2 (threaded to L3 or ret)
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        jump_targets = [l.split()[-1] for l in lines if l.startswith(("jp ", "jr "))]
        # All jumps should target L3 or be eliminated
        for target in jump_targets:
            assert target in ("L3", "ret"), f"Unexpected jump target: {target}"


class TestRelativeJumps:
    """Test jp to jr conversion."""

    def test_jp_to_jr_nearby_label(self):
        """jp LABEL -> jr LABEL when target is close."""
        asm = "LOOP:\n\tnop\n\tjp LOOP"
        result = optimize(asm)
        assert "jr LOOP" in result or "djnz" in result

    def test_jp_conditional_to_jr(self):
        """jp z,LABEL -> jr z,LABEL when close."""
        asm = "TARGET:\n\tnop\n\tor a\n\tjp z,TARGET"
        result = optimize(asm)
        assert "jr z,TARGET" in result


class TestDeadStoreElimination:
    """Test dead store elimination at procedure entry."""

    def test_dead_store_removed(self):
        """Store to unused memory location is removed."""
        asm = "myproc:\n\tld (PARAM),a\n\tadd a,b\n\tret\nPARAM:\tds 1"
        result = optimize(asm)
        assert "myproc:" in result
        assert "(PARAM)" not in result
        # One this module does not define is another's.
        assert "(PARAM)" in optimize("myproc:\n\tld (PARAM),a\n\tadd a,b\n\tret")

    def test_live_store_kept(self):
        """Store to memory location that is later loaded is kept."""
        asm = "myproc:\n\tld (PARAM),a\n\tcall other\n\tld a,(PARAM)\n\tret"
        result = optimize(asm)
        assert "(PARAM)" in result

    def test_store_after_internal_label_is_kept(self):
        """A `??' label is a join point mid-expression, not a procedure entry.

        `var = (a = b)' in PL/M-80 compiles to the two arms of a comparison
        meeting at a generated label, with the store to var right after it.
        Treating that label as a procedure entry threw the assignment away.
        """
        asm = ("P3:\n\tld a,(VA)\n\tcp b\n\tjr z,??T\n\txor a\n\tjr ??C\n"
               "??T:\n\tld a,0ffh\n??C:\n\tld (S),a\n\tret\n")
        result = optimize(asm)
        assert "ld\t(S),a" in result or "ld (S),a" in result, result

    def test_store_read_by_a_later_procedure_is_kept(self):
        """Liveness is a property of the module, not of one procedure.

        Scanning only to the end of the procedure is right for a parameter
        slot, which nothing else can name, and wrong for a variable at module
        scope that a procedure declared further down reads.
        """
        asm = "P5:\n\tld (V),a\n\tret\nP6:\n\tld a,(V)\n\tret\n"
        result = optimize(asm)
        assert "(V),a" in result, result

    def test_exported_store_is_kept(self):
        """A location the module exports may be read by another module."""
        asm = "\tpublic PARAM\nmyproc:\n\tld (PARAM),a\n\tadd a,b\n\tret"
        result = optimize(asm)
        assert "(PARAM)" in result, result

    def test_store_whose_address_is_taken_is_kept(self):
        """`ld hl,PARAM' hands the location to code this pass cannot see."""
        asm = "myproc:\n\tld (PARAM),a\n\tld hl,PARAM\n\tcall other\n\tret"
        result = optimize(asm)
        assert "(PARAM)" in result, result


class TestOutputIndentation:
    """Test that output uses consistent indentation."""

    def test_pattern_replacement_uses_tabs(self):
        """Pattern replacements should use tab indentation."""
        result = optimize("\tld a,0\n\tcp b")
        # Should be tab-indented xor a
        assert "\txor a" in result

    def test_push_pop_replacement_uses_tabs(self):
        """Push/pop copy replacements should use tab indentation."""
        result = optimize("\tpush hl\n\tpop de")
        for line in result.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.endswith(":"):
                assert line.startswith("\t"), f"Line not tab-indented: {repr(line)}"


class TestVersionConsistency:
    """Test that version numbers are consistent."""

    def test_version_matches(self):
        """__init__.py version should match installed package metadata (pyproject.toml)."""
        import upeepz80
        from importlib.metadata import version
        assert upeepz80.__version__ == version("upeepz80")
