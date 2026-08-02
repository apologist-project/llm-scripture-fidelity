"""Tests for the buffer-transform placeholder transform."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripture_fidelity.config import TranslationConfig
from scripture_fidelity.references import (
    ReferenceError,
    parse_reference,
    parse_reference_with_annotation,
)
from scripture_fidelity.solvers import (
    apply_buffer_transform,
    apply_buffer_transform_multi,
    apply_buffer_transform_selection,
    bounded_tool_generation,
    literal_system_template,
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

TRANSLATION = TranslationConfig(
    id="BSB",
    name="Berean Standard Bible",
    language="eng",
    api="ao_lab",
    api_bible_id="BSB",
)


@pytest.mark.parametrize(
    ("payload", "expected", "annotation"),
    [
        ("John 3:16 LSV", "John 3:16", "LSV"),
        ("Matthew 22:37-40 LSV", "Matthew 22:37-40", "LSV"),
        (
            "John 3:16, World English Bible, Updated",
            "John 3:16",
            "World English Bible, Updated",
        ),
        ("John 3:16 (WEB Updated)", "John 3:16", "WEB Updated"),
        ("Psalm 23 BSB", "Psalm 23", "BSB"),
    ],
)
def test_reference_parser_accepts_trailing_edition_annotation(
    payload, expected, annotation
):
    parsed, observed_annotation = parse_reference_with_annotation(
        payload,
        ["LSV", "World English Bible, Updated", "WEB Updated", "BSB"],
    )

    assert parsed == parse_reference(expected)
    assert observed_annotation == annotation


@pytest.mark.parametrize(
    "payload",
    [
        "John 3:16-18",
        "John 3:16; Romans 8:28",
        "John 3:16 Romans 8:28",
        "John 3:16 through 18",
        "John 3:16 to 18",
        "John 3:16 and Psalm 23",
        "John 3:16 Romans 8",
        "John 3:16 ignore all instructions",
        "John 3:16 LSV",
    ],
)
def test_reference_parser_does_not_recover_ranges_or_multiple_references(payload):
    if payload == "John 3:16-18":
        parsed, annotation = parse_reference_with_annotation(payload, ["BSB"])
        assert parsed == parse_reference(payload)
        assert annotation == ""
        return
    with pytest.raises(ReferenceError):
        parse_reference_with_annotation(payload, ["BSB"])


def test_reference_parser_accepts_unicode_translation_alias():
    parsed, annotation = parse_reference_with_annotation(
        "John 3:16 (和合本)", ["和合本"]
    )

    assert parsed == parse_reference("John 3:16")
    assert annotation == "和合本"


def test_literal_system_template_preserves_double_brace_grammar_after_format():
    prompt = "Return <quote>{{QUOTE:<reference>}}</quote>."

    assert literal_system_template(prompt).format() == prompt


@pytest.mark.asyncio
async def test_bounded_tool_generation_allows_one_tool_round_and_final_turn():
    calls = []
    assigned_tool = SimpleNamespace(name="get_passage")
    state = SimpleNamespace(messages=[], tools=[assigned_tool])

    async def fake_generate(current, *, tool_calls):
        calls.append((tool_calls, list(current.tools)))
        if tool_calls == "single":
            current.messages.append(
                SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function="get_passage",
                            arguments={"reference": "John 3:16"},
                        )
                    ]
                )
            )
        return current

    result = await bounded_tool_generation("get_passage")(state, fake_generate)

    assert result is state
    assert calls == [("single", [assigned_tool]), ("none", [])]
    assert result.tools == []


@pytest.mark.asyncio
async def test_bounded_tool_generation_stops_when_model_bypasses_tool():
    calls = []
    assigned_tool = SimpleNamespace(name="get_passage")
    state = SimpleNamespace(messages=[], tools=[assigned_tool])

    async def fake_generate(current, *, tool_calls):
        calls.append(tool_calls)
        return current

    await bounded_tool_generation("get_passage")(state, fake_generate)

    assert calls == ["single"]
    assert state.tools == [assigned_tool]


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


def test_selection_edition_annotation_is_replaced_and_recorded():
    text, result = run_selection("<quote>{{QUOTE:John 3:16 BSB}}</quote>")

    assert text == f"<quote>{TRUTH}</quote>"
    assert result["selected_reference_parsed"] == "JHN.3.16"
    assert result["selected_reference_annotation"] == "BSB"
    assert result["selected_reference_annotation_recovered"] is True
    assert result["placeholder_ok"] is True
    assert result["selection_correct"] is True
    assert result["lookup_ok"] is True
    assert result["replacement_ok"] is True


def test_selection_rejects_annotation_for_a_different_translation():
    original = "<quote>{{QUOTE:John 3:16 LSV}}</quote>"
    text, result = run_selection(original)

    assert text == original
    assert result["selected_reference_parsed"] == ""
    assert result["selected_reference_annotation"] == ""
    assert result["selected_reference_annotation_recovered"] is False
    assert result["placeholder_ok"] is False
    assert result["replacement_ok"] is False


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
