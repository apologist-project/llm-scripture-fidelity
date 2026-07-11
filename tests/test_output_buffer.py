"""Tests for the output-buffer placeholder transform."""

from scripture_fidelity.solvers import apply_output_buffer, apply_output_buffer_multi

TRUTH = "For God so loved the world..."
TRUTH2 = "Praise the LORD, all you nations!"


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


def test_multi_replaces_all_placeholders():
    expected = [("John 3:16", TRUTH), ("Psalm 117", TRUTH2)]
    text, ok = apply_output_buffer_multi(
        '<quote ref="John 3:16">{{QUOTE:John 3:16}}</quote>\n'
        '<quote ref="Psalm 117">{{QUOTE:Psalm 117}}</quote>',
        expected,
    )
    assert TRUTH in text and TRUTH2 in text
    assert ok is True


def test_multi_missing_placeholder_not_ok():
    expected = [("John 3:16", TRUTH), ("Psalm 117", TRUTH2)]
    text, ok = apply_output_buffer_multi("{{QUOTE:John 3:16}}", expected)
    assert text == TRUTH
    assert ok is False


def test_multi_duplicate_placeholder_not_ok():
    expected = [("John 3:16", TRUTH), ("Psalm 117", TRUTH2)]
    text, ok = apply_output_buffer_multi(
        "{{QUOTE:John 3:16}} {{QUOTE:John 3:16}}", expected
    )
    assert ok is False


def test_multi_alias_reference_matches():
    expected = [("John 3:16", TRUTH)]
    text, ok = apply_output_buffer_multi("{{QUOTE:Jn 3:16}}", expected)
    assert text == TRUTH
    assert ok is True
