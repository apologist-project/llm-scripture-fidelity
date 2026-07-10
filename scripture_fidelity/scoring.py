"""Deterministic quotation-fidelity metrics and the Inspect scorer."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

QUOTE_RE = re.compile(r"<quote>(.*?)</quote>", re.DOTALL | re.IGNORECASE)

# Characters normalized to ASCII equivalents before comparison
_CHAR_MAP = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
        "\u00a0": " ", "\u2009": " ", "\u200a": " ", "\u2028": " ",
    }
)


def extract_quote(completion: str) -> str:
    """Extract the quoted passage from a model completion.

    Prefers the last <quote>...</quote> block; falls back to the whole
    completion (stripped) when no tags are present.
    """
    matches = QUOTE_RE.findall(completion or "")
    if matches:
        return matches[-1].strip()
    return (completion or "").strip()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str, language: str = "eng") -> str:
    """Normalize for lenient comparison: NFKC, unified quotes/dashes,
    casefolded, punctuation removed, whitespace collapsed (removed for zh)."""
    text = unicodedata.normalize("NFKC", text or "").translate(_CHAR_MAP)
    text = text.casefold()
    # Drop punctuation and symbol characters (covers CJK punctuation too)
    text = "".join(
        c for c in text if not unicodedata.category(c).startswith(("P", "S"))
    )
    text = collapse_ws(text)
    if language.startswith(("zh", "zho", "cmn")):
        text = re.sub(r"\s+", "", text)
    return text


def compute_metrics(
    answer: str,
    ground_truth: str,
    ground_truth_verses: list[str],
    language: str = "eng",
) -> dict[str, float]:
    """Compare an extracted quote against the ground-truth passage text."""
    norm_answer = normalize(answer, language)
    norm_truth = normalize(ground_truth, language)

    exact = 1.0 if collapse_ws(answer) == collapse_ws(ground_truth) else 0.0
    normalized_match = 1.0 if norm_answer == norm_truth else 0.0
    similarity = fuzz.ratio(norm_answer, norm_truth) / 100.0 if norm_truth else 0.0
    cer = (
        Levenshtein.normalized_distance(norm_truth, norm_answer)
        if norm_truth
        else 1.0
    )

    norm_verses = [normalize(v, language) for v in ground_truth_verses]
    norm_verses = [v for v in norm_verses if v]
    if norm_verses:
        covered = sum(1 for v in norm_verses if v in norm_answer)
        verse_coverage = covered / len(norm_verses)
    else:
        verse_coverage = 0.0

    return {
        "exact": exact,
        "normalized": normalized_match,
        "similarity": round(similarity, 4),
        "cer": round(cer, 4),
        "verse_coverage": round(verse_coverage, 4),
        "answered": 1.0 if norm_answer else 0.0,
    }


METRIC_KEYS = [
    "exact",
    "normalized",
    "similarity",
    "cer",
    "verse_coverage",
    "answered",
    "placeholder_ok",
]


def quotation_fidelity():
    """Inspect scorer producing the full metric dict per sample.

    ``placeholder_ok`` is only meaningful for the output_buffer method
    (whether the model emitted a well-formed, correctly-referenced
    placeholder); it is fixed at 1.0 for other methods.
    """
    from inspect_ai.scorer import Score, Target, mean, scorer, stderr
    from inspect_ai.solver import TaskState

    @scorer(
        metrics={key: [mean(), stderr()] for key in METRIC_KEYS},
        name="quotation_fidelity",
    )
    def _quotation_fidelity():
        async def score(state: TaskState, target: Target) -> Score:
            quote = extract_quote(state.output.completion)
            language = state.metadata.get("text_language", "eng")
            verses = state.metadata.get("ground_truth_verses", [])
            metrics = compute_metrics(quote, target.text, verses, language)
            if state.metadata.get("method") == "output_buffer":
                metrics["placeholder_ok"] = (
                    1.0 if state.store.get("placeholder_ok") else 0.0
                )
            else:
                metrics["placeholder_ok"] = 1.0
            return Score(
                value=metrics,
                answer=quote,
                explanation=(
                    f"similarity={metrics['similarity']:.3f} "
                    f"cer={metrics['cer']:.3f} exact={int(metrics['exact'])}"
                ),
            )

        return score

    return _quotation_fidelity()
