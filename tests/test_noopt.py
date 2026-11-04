""";#noopt regions and PeepholeOptimizer(disable=...) (um80_and_friends
docs/debug_info.md; cpmemu docs/debugger_plan.md, D6)."""

from upeepz80 import PeepholeOptimizer, optimize

# Each text has a rewrite in it: tail call, push/pop pair, jp to the next
# label, jp that reaches (jr), dead store at entry.
PLAIN_TAIL = "F:\n\tcall G\n\tret\n\tnop\n\tnop\nG:\n\tret\n"
BODY = ("\tld hl,5\n\tpush hl\n\tpop de\n\tcall G\n\tret\n"
        "\tjp N\nN:\n\tjp M\nM:\n\txor a\n\tret\n")


def region(body: str) -> str:
    return ";#noopt begin\n" + body + ";#noopt end\n"


def test_a_text_with_no_noopt_is_not_changed():
    src = "F:\n" + BODY + "G:\n\tret\n"
    assert optimize(src) == optimize(src + ";#line 1 1\n").replace(";#line 1 1\n", "")


def test_the_body_does_change_without_the_region():
    out = optimize("F:\n" + BODY + "G:\n\tret\n")
    assert out != "F:\n" + BODY + "G:\n\tret\n"
    assert "jp G" in out or "jr G" in out


def test_a_region_is_kept_to_the_letter():
    src = "F:\n" + region(BODY) + "G:\n\tret\n"
    assert optimize(src) == src


def test_a_region_keeps_call_and_ret():
    src = "F:\n;#noopt begin\n\tcall G\n\tret\n;#noopt end\nG:\n\tret\n"
    assert optimize(src) == src
    plain = optimize(PLAIN_TAIL)
    assert "call G" not in plain and "jr G" in plain


def test_code_outside_the_region_still_changes():
    src = "H:\n\tpush hl\n\tpop de\n\tret\n" + "F:\n" + region(BODY) + "G:\n\tret\n"
    out = optimize(src)
    assert region(BODY) in out
    assert "push hl" not in out.split("F:")[0]


def test_a_value_read_in_the_region_is_not_dead():
    """B is loaded, then the region reads it: the store before the region
    stays.  (Outside a region, the load of B that is overwritten is dead.)"""
    src = ("F:\n\tld a,1\n\tld b,a\n" + region("\tld a,b\n\tld (V),a\n\tret\n")
           + "V:\tds 1\n")
    out = optimize(src)
    assert "ld b,a" in out
    # the same code with no region: ld b,a / ld a,b is shortened
    plain = optimize("F:\n\tld a,1\n\tld b,a\n\tld a,b\n\tld (V),a\n\tret\nV:\tds 1\n")
    assert plain.count("ld b,a") <= out.count("ld b,a")


def test_a_label_in_the_region_that_outside_code_jumps_to_stays():
    src = ("F:\n\tjp T\n\tnop\n" + region("T:\n\tjp U\n") + "U:\n\tret\n")
    out = optimize(src)
    assert "T:\n\tjp U\n" in out
    assert ";#noopt begin" in out and ";#noopt end" in out


def test_a_jp_in_the_region_does_not_become_jr():
    src = "F:\n" + region("\tjp LB\n\tnop\nLB:\n\tret\n")
    assert optimize(src) == src
    assert "jr LB" in optimize("F:\n\tjp LB\n\tnop\nLB:\n\tret\n")


def test_nested_regions_and_a_missing_end():
    src = ("F:\n;#noopt begin\n;#noopt begin\n\tcall G\n;#noopt end\n\tret\n"
           ";#noopt end\nG:\n\tret\n")
    assert optimize(src) == src
    open_src = "F:\n;#noopt begin\n\tcall G\n\tret\nG:\n\tret\n"
    assert optimize(open_src) == open_src


def test_an_end_with_no_begin_is_ignored():
    src = "F:\n;#noopt end\n\tcall G\n\tret\nG:\n\tret\n"
    assert "call G" not in optimize(src)


def test_a_marker_in_the_region_is_kept():
    src = "F:\n;#noopt begin\n;#line 1 3\n\tcall G\n;#line 1 4\n\tret\n;#noopt end\nG:\n\tret\n"
    assert optimize(src) == src


def test_disable_tail_call():
    assert "call G\n\tret" not in optimize(PLAIN_TAIL)
    out = PeepholeOptimizer(disable={"tail_call"}).optimize(PLAIN_TAIL)
    assert "call G\n\tret" in out
    assert optimize(PLAIN_TAIL, disable={"tail_call"}) == out


def test_disable_names_other_rewrites():
    src = "F:\n\tjp LB\n\tnop\nLB:\n\tret\n"
    assert "jr LB" in optimize(src)
    assert optimize(src, disable={"jr_convert"}) == src
    src2 = "F:\n\tjp LA\n\tnop\nLA:\n\tjp LB\n\tnop\nLB:\n\tret\n"
    assert "jp LA" in optimize(src2, disable={"jump_thread", "jr_convert"})
    assert "LA" not in optimize(src2).split("LA:")[0]


def test_no_disable_gives_the_same_output():
    src = "F:\n" + BODY + "G:\n\tret\n"
    assert PeepholeOptimizer(disable=set()).optimize(src) == optimize(src)


def test_disable_a_z80_rewrite():
    from .test_peephole import KILL_ALL_BUT_HL
    src = "\tld de,0\n\tcall ??subde" + KILL_ALL_BUT_HL
    assert "subde" not in optimize(src)
    assert "call ??subde" in optimize(src, disable={"subde_zero"})


def test_stats_do_not_count_a_disabled_pattern():
    opt = PeepholeOptimizer(disable={"tail_call"})
    opt.optimize(PLAIN_TAIL)
    assert "tail_call" not in opt.stats
