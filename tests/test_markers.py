"""``;#`` debug markers (um80_and_friends docs/debug_info.md).

``conftest.py`` runs every ``optimize()`` call of the other tests a second
time with markers added, and checks that the markers stay in order and that
the instructions do not change.  The tests here name the cases.
"""

from upeepz80 import optimize

from .conftest import add_markers, markers_of, without_markers


def test_a_fused_pair_lands_after_the_markers_it_passed():
    out = optimize("\tld hl,5\n;#line 1 6\n\tpush hl\n;#line 1 7\n\tpop de\n")
    assert markers_of(out) == [";#line 1 6", ";#line 1 7"]
    assert without_markers(out).split() == ["ld", "hl,5", "ld", "d,h", "ld", "e,l"]


def test_a_marker_in_front_of_a_dead_store_does_not_hide_the_store():
    src = "P:\n;#line 1 3\n\tld (X),a\n;#line 1 4\n\tadd a,b\n\tret\nX:\tds 1\n"
    out = optimize(src)
    assert "ld (X),a" not in out
    assert markers_of(out) == [";#line 1 3", ";#line 1 4"]


def test_a_plain_comment_in_front_of_a_store_still_hides_the_store():
    """The text of other compilers with a comment there gives the output of
    0.2.7: only a marker line is skipped."""
    src = "P:\n; keep\n\tld (X),a\n\tadd a,b\n\tret\nX:\tds 1\n"
    assert "ld (X),a" in optimize(src)


def test_a_marker_of_deleted_dead_code_goes_in_front_of_the_next_marker():
    src = ("\tjp L\n;#line 1 2\n\tld a,1\n;#line 1 3\n\tld b,2\nL:\n"
           ";#line 1 4\n\tret\n")
    out = optimize(src)
    assert markers_of(out) == [";#line 1 2", ";#line 1 3", ";#line 1 4"]
    assert without_markers(out).split() == without_markers(
        optimize(without_markers(src))).split()
    # the lost markers stand in front of the marker that is left
    assert out.index(";#line 1 3") < out.index(";#line 1 4")


def test_each_marker_of_a_marked_text_is_kept_in_order():
    src = ("f:\n\tcall g\n\tret\ng:\n\tld a,(V)\n\tor a\n\tjr z,h\n"
           "\tinc a\nh:\n\tld (V),a\n\tret\nV:\tds 1\n")
    marked = add_markers(src)
    out = optimize(marked)
    assert markers_of(out) == markers_of(marked)
    assert without_markers(out) == optimize(src)


def test_a_marker_with_blanks_is_a_marker():
    out = optimize("\tld a,0\n;  #line 1 2\n\tcp b\n")
    assert markers_of(out) == [";  #line 1 2"]
