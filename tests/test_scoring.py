"""Tests for quote extraction, normalization, and fidelity metrics."""

from scripture_fidelity.scoring import compute_metrics, extract_quote, normalize

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
