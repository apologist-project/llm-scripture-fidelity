"""Tests for the buffer-transform placeholder transform."""

import asyncio

from scripture_fidelity.solvers import (
    apply_buffer_transform,
    apply_buffer_transform_multi,
    apply_buffer_transform_selection,
)

TRUTH = "For God so loved the world..."
TRUTH2 = "Praise the LORD, all you nations!"


def test_replaces_matching_placeholder():
    text, ok = apply_buffer_transform(
        "<quote>{{QUOTE:John 3:16}}</quote>", "John 3:16", TRUTH
    )
    assert text == f"<quote>{TRUTH}</quote>"
    assert ok is True


def test_placeholder_reference_alias_matches():
    text, ok = apply_buffer_transform("{{QUOTE:Jn 3:16}}", "John 3:16", TRUTH)
    assert text == TRUTH
    assert ok is True


def test_whitespace_inside_placeholder():
    text, ok = apply_buffer_transform("{{ QUOTE : John 3:16 }}", "John 3:16", TRUTH)
    assert text == TRUTH
    assert ok is True


def test_wrong_reference_left_in_place():
    original = "{{QUOTE:Genesis 1:1}}"
    text, ok = apply_buffer_transform(original, "John 3:16", TRUTH)
    assert text == original
    assert ok is False


def test_no_placeholder():
    text, ok = apply_buffer_transform("I cannot do that.", "John 3:16", TRUTH)
    assert text == "I cannot do that."
    assert ok is False


def test_multiple_placeholders_not_ok():
    text, ok = apply_buffer_transform(
        "{{QUOTE:John 3:16}} {{QUOTE:John 3:16}}", "John 3:16", TRUTH
    )
    assert text == f"{TRUTH} {TRUTH}"
    assert ok is False


def test_unparseable_placeholder_reference():
    original = "{{QUOTE:NotABook 99}}"
    text, ok = apply_buffer_transform(original, "John 3:16", TRUTH)
    assert text == original
    assert ok is False


def test_multi_replaces_all_placeholders():
    expected = [("John 3:16", TRUTH), ("Psalm 117", TRUTH2)]
    text, ok = apply_buffer_transform_multi(
        '<quote ref="John 3:16">{{QUOTE:John 3:16}}</quote>\n'
        '<quote ref="Psalm 117">{{QUOTE:Psalm 117}}</quote>',
        expected,
    )
    assert TRUTH in text and TRUTH2 in text
    assert ok is True


def test_multi_missing_placeholder_not_ok():
    expected = [("John 3:16", TRUTH), ("Psalm 117", TRUTH2)]
    text, ok = apply_buffer_transform_multi("{{QUOTE:John 3:16}}", expected)
    assert text == TRUTH
    assert ok is False


def test_multi_duplicate_placeholder_not_ok():
    expected = [("John 3:16", TRUTH), ("Psalm 117", TRUTH2)]
    text, ok = apply_buffer_transform_multi(
        "{{QUOTE:John 3:16}} {{QUOTE:John 3:16}}", expected
    )
    assert ok is False


def test_multi_alias_reference_matches():
    expected = [("John 3:16", TRUTH)]
    text, ok = apply_buffer_transform_multi("{{QUOTE:Jn 3:16}}", expected)
    assert text == TRUTH
    assert ok is True


# --- buffer_transform_selection ---------------------------------------------

from dataclasses import dataclass

from scripture_fidelity.config import TranslationConfig
from scripture_fidelity.references import parse_reference

TRANSLATION = TranslationConfig(
    id="BSB", language="eng", api="ao_lab", api_bible_id="BSB"
)

# Fixture texts keyed by canonical USFM — the lookup only knows these
FIXTURES = {
    parse_reference("John 3:16").usfm(): TRUTH,
    parse_reference("Psalm 117").usfm(): TRUTH2,
}


@dataclass
class _Passage:
    text: str


async def fake_lookup(translation, parsed):
    text = FIXTURES.get(parsed.usfm())
    if text is None:
        raise LookupError(parsed.usfm())
    return _Passage(text)


def run_selection(completion, expected_ref="John 3:16"):
    return asyncio.run(
        apply_buffer_transform_selection(
            completion, expected_ref, TRANSLATION, fake_lookup
        )
    )


def test_selection_correct_and_replaced():
    text, r = run_selection("<quote>{{QUOTE:John 3:16}}</quote>")
    assert text == f"<quote>{TRUTH}</quote>"
    assert r["placeholder_ok"] is True
    assert r["selection_correct"] is True
    assert r["lookup_ok"] is True
    assert r["replacement_ok"] is True
    assert r["lookup_fixture_id"] == f"{TRANSLATION.source_key}:JHN.3.16"


def test_wrong_selection_gets_wrong_passage_not_expected_text():
    text, r = run_selection("{{QUOTE:Psalm 117}}", expected_ref="John 3:16")
    assert text == TRUTH2  # replaced with the *selected* passage's text
    assert TRUTH not in text  # never the scenario's expected text
    assert r["selection_correct"] is False
    assert r["lookup_ok"] is True
    assert r["replacement_ok"] is True


def test_selection_malformed_reference():
    original = "{{QUOTE:NotABook 99}}"
    text, r = run_selection(original)
    assert text == original
    assert r["placeholder_ok"] is False
    assert r["selection_correct"] is False
    assert r["lookup_ok"] is False
    assert r["replacement_ok"] is False


def test_selection_missing_placeholder():
    text, r = run_selection("I think it's John 3:16.")
    assert text == "I think it's John 3:16."
    assert r["placeholder_count"] == 0
    assert r["placeholder_ok"] is False
    assert r["replacement_ok"] is False


def test_selection_duplicate_placeholders_not_replaced():
    original = "{{QUOTE:John 3:16}} {{QUOTE:John 3:16}}"
    text, r = run_selection(original)
    assert text == original
    assert r["placeholder_count"] == 2
    assert r["placeholder_ok"] is False
    assert r["replacement_ok"] is False
    # the correct selection is still observed independently
    assert r["selection_correct"] is True


def test_selection_lookup_failure_leaves_placeholder():
    text, r = run_selection("{{QUOTE:Genesis 1:1}}")
    assert text == "{{QUOTE:Genesis 1:1}}"
    assert r["placeholder_ok"] is True
    assert r["selection_correct"] is False
    assert r["lookup_ok"] is False
    assert r["replacement_ok"] is False


def test_selection_alias_counts_as_correct():
    text, r = run_selection("{{QUOTE:Jn 3:16}}")
    assert text == TRUTH
    assert r["selection_correct"] is True
    assert r["selected_reference_parsed"] == "JHN.3.16"
