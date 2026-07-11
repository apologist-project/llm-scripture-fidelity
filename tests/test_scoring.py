"""Tests for quote extraction, normalization, and fidelity metrics."""

from scripture_fidelity.scoring import (
    compute_metrics,
    compute_multi_metrics,
    extract_quote,
    extract_quotes,
    fail_fidelity,
    normalize,
    tool_coverage,
    tool_was_used,
)

BSB_JOHN_3_16 = (
    "For God so loved the world that He gave His one and only Son, "
    "that everyone who believes in Him shall not perish but have eternal life."
)


def test_extract_quote_tags():
    completion = f"Sure, here it is:\n<quote>{BSB_JOHN_3_16}</quote>"
    assert extract_quote(completion) == BSB_JOHN_3_16


def test_extract_quote_prefers_last_block():
    completion = "<quote>draft</quote> hmm <quote>final</quote>"
    assert extract_quote(completion) == "final"


def test_extract_quote_fallback_without_tags():
    assert extract_quote(f"  {BSB_JOHN_3_16}  ") == BSB_JOHN_3_16


def test_normalize_unifies_quotes_case_punct():
    assert normalize("\u201cHe said, \u2018Go!\u2019\u201d") == normalize(
        '"he said, \'go\'"'
    )


def test_normalize_chinese_removes_spaces():
    assert normalize("\u795e \u7231 \u4e16\u4eba", "zho") == "\u795e\u7231\u4e16\u4eba"


def test_exact_match():
    m = compute_metrics(BSB_JOHN_3_16, BSB_JOHN_3_16, [BSB_JOHN_3_16])
    assert m["exact"] == 1.0
    assert m["normalized"] == 1.0
    assert m["similarity"] == 1.0
    assert m["cer"] == 0.0
    assert m["verse_coverage"] == 1.0
    assert m["answered"] == 1.0


def test_normalized_but_not_exact():
    answer = BSB_JOHN_3_16.replace('"', "\u201c").upper()
    m = compute_metrics(answer, BSB_JOHN_3_16, [BSB_JOHN_3_16])
    assert m["exact"] == 0.0
    assert m["normalized"] == 1.0


def test_near_miss_scores_high_similarity():
    answer = BSB_JOHN_3_16.replace("everyone", "whoever")
    m = compute_metrics(answer, BSB_JOHN_3_16, [BSB_JOHN_3_16])
    assert m["exact"] == 0.0
    assert m["normalized"] == 0.0
    assert m["similarity"] > 0.9
    assert 0.0 < m["cer"] < 0.2
    assert m["verse_coverage"] == 0.0


def test_empty_answer():
    m = compute_metrics("", BSB_JOHN_3_16, [BSB_JOHN_3_16])
    assert m["answered"] == 0.0
    assert m["similarity"] == 0.0
    assert m["cer"] == 1.0


def test_verse_coverage_partial():
    v1 = "In the beginning God created the heavens and the earth."
    v2 = "Now the earth was formless and void."
    m = compute_metrics(v1, f"{v1} {v2}", [v1, v2])
    assert m["verse_coverage"] == 0.5


def test_tool_was_used():
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
    from inspect_ai.tool import ToolCall

    call = ToolCall(
        id="1", function="get_passage", arguments={"reference": "John 3:16"}
    )
    messages = [
        ChatMessageUser(content="Quote John 3:16"),
        ChatMessageAssistant(content="", tool_calls=[call]),
        ChatMessageAssistant(content="<quote>...</quote>"),
    ]
    assert tool_was_used(messages, "get_passage") is True
    assert tool_was_used(messages, "search_web") is False
    assert tool_was_used([], "get_passage") is False


def test_fail_fidelity_zeroes_scores():
    m = compute_metrics(BSB_JOHN_3_16, BSB_JOHN_3_16, [BSB_JOHN_3_16])
    m["tool_used"] = 0.0
    fail_fidelity(m)
    assert m["exact"] == 0.0
    assert m["normalized"] == 0.0
    assert m["similarity"] == 0.0
    assert m["verse_coverage"] == 0.0
    assert m["cer"] == 1.0
    assert m["answered"] == 1.0  # answered is preserved


def test_extract_quotes_attributed():
    completion = (
        '<quote ref="Psalm 117">praise</quote>\n'
        '<quote ref="John 3:16">love</quote>'
    )
    quotes = extract_quotes(completion, ["John 3:16", "Psalm 117"])
    assert quotes == {"John 3:16": "love", "Psalm 117": "praise"}


def test_extract_quotes_ref_alias_matches():
    quotes = extract_quotes('<quote ref="Jn 3:16">love</quote>', ["John 3:16"])
    assert quotes == {"John 3:16": "love"}


def test_extract_quotes_positional_fallback():
    completion = "<quote>first</quote> <quote>second</quote>"
    quotes = extract_quotes(completion, ["John 3:16", "Psalm 117"])
    assert quotes == {"John 3:16": "first", "Psalm 117": "second"}


def test_extract_quotes_missing_reference_is_empty():
    quotes = extract_quotes(
        '<quote ref="John 3:16">love</quote>', ["John 3:16", "Psalm 117"]
    )
    assert quotes == {"John 3:16": "love", "Psalm 117": ""}


def test_compute_multi_metrics_averages_and_penalizes_missing():
    v1 = "In the beginning God created the heavens and the earth."
    m = compute_multi_metrics(
        {"Genesis 1:1": v1, "John 3:16": ""},
        [v1, BSB_JOHN_3_16],
        [[v1], [BSB_JOHN_3_16]],
        ["Genesis 1:1", "John 3:16"],
    )
    assert m["exact"] == 0.5
    assert m["verse_coverage"] == 0.5
    assert m["answered"] == 0.5
    assert 0.4 < m["cer"] <= 0.5001


def test_tool_coverage_fraction():
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCall

    calls = [
        ToolCall(id="1", function="get_passage", arguments={"reference": "Jn 3:16"}),
        ToolCall(id="2", function="get_passage", arguments={"reference": "Acts 2:1"}),
    ]
    messages = [ChatMessageAssistant(content="", tool_calls=calls)]
    refs = ["John 3:16", "Psalm 117", "Genesis 1:1"]
    assert tool_coverage(messages, "get_passage", refs) == 1 / 3
    assert tool_coverage(messages, "get_passage", ["John 3:16"]) == 1.0
    assert tool_coverage([], "get_passage", refs) == 0.0
