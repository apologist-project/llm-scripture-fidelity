"""Tests for the output-buffer placeholder transform."""

from scripture_fidelity.solvers import apply_output_buffer

TRUTH = "For God so loved the world..."


def test_replaces_matching_placeholder():
    text, ok = apply_output_buffer(
        "<quote>{{QUOTE:John 3:16}}</quote>", "John 3:16", TRUTH
    )
    assert text == f"<quote>{TRUTH}</quote>"
    assert ok is True


def test_placeholder_reference_alias_matches():
    text, ok = apply_output_buffer("{{QUOTE:Jn 3:16}}", "John 3:16", TRUTH)
    assert text == TRUTH
    assert ok is True


def test_whitespace_inside_placeholder():
    text, ok = apply_output_buffer("{{ QUOTE : John 3:16 }}", "John 3:16", TRUTH)
    assert text == TRUTH
    assert ok is True


def test_wrong_reference_left_in_place():
    original = "{{QUOTE:Genesis 1:1}}"
    text, ok = apply_output_buffer(original, "John 3:16", TRUTH)
    assert text == original
    assert ok is False


def test_no_placeholder():
    text, ok = apply_output_buffer("I cannot do that.", "John 3:16", TRUTH)
    assert text == "I cannot do that."
    assert ok is False


def test_multiple_placeholders_not_ok():
    text, ok = apply_output_buffer(
        "{{QUOTE:John 3:16}} {{QUOTE:John 3:16}}", "John 3:16", TRUTH
    )
    assert text == f"{TRUTH} {TRUTH}"
    assert ok is False


def test_unparseable_placeholder_reference():
    original = "{{QUOTE:NotABook 99}}"
    text, ok = apply_output_buffer(original, "John 3:16", TRUTH)
    assert text == original
    assert ok is False
