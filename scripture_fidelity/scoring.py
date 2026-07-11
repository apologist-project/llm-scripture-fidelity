"""Deterministic quotation-fidelity metrics and the Inspect scorer."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

QUOTE_RE = re.compile(r"<quote>(.*?)</quote>", re.DOTALL | re.IGNORECASE)
QUOTE_ATTR_RE = re.compile(
    r"<quote(?:\s+ref\s*=\s*(?:\"([^\"]*)\"|'([^']*)'))?\s*>(.*?)</quote>",
    re.DOTALL | re.IGNORECASE,
)

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


def extract_quotes(completion: str, references: list[str]) -> dict[str, str]:
    """Extract one quote per requested reference from a multi-reference
    completion.

    Quote blocks carrying a matching ``ref`` attribute (compared via
    parse_reference, so aliases match) are attributed to their reference;
    remaining unattributed blocks are assigned positionally to the
    references still missing a quote. Missing references map to "".
    """
    from scripture_fidelity.references import ReferenceError, parse_reference

    parsed = {}
    for ref_str in references:
        try:
            parsed[ref_str] = parse_reference(ref_str)
        except ReferenceError:
            parsed[ref_str] = None

    quotes: dict[str, str] = {ref: "" for ref in references}
    unattributed: list[str] = []
    for m in QUOTE_ATTR_RE.finditer(completion or ""):
        attr = m.group(1) or m.group(2)
        text = m.group(3).strip()
        target = None
        if attr:
            try:
                attr_parsed = parse_reference(attr)
                target = next(
                    (r for r in references if parsed[r] == attr_parsed), None
                )
            except ReferenceError:
                target = None
        if target is not None:
            quotes[target] = text
        else:
            unattributed.append(text)

    remaining = iter(unattributed)
    for ref in references:
        if not quotes[ref]:
            quotes[ref] = next(remaining, "")
    return quotes


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
    "tool_used",
]

# Tool each method is instructed to use (None = no tool expected)
METHOD_TOOLS = {"tool_call": "get_passage", "web_search": "search_web"}

# Fidelity metrics invalidated when the model disobeyed the method's
# instructions (quoting from memory must not count as method fidelity)
_FIDELITY_KEYS = ("exact", "normalized", "similarity", "verse_coverage")


def tool_was_used(messages, tool_name: str) -> bool:
    """True if any assistant message in the transcript called ``tool_name``."""
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            if call.function == tool_name:
                return True
    return False


def tool_coverage(messages, tool_name: str, references: list[str]) -> float:
    """Fraction of requested references that were looked up via ``tool_name``.

    Each tool call's ``reference`` argument is matched against the requested
    references with parse_reference equality (so aliases like "Jn 3:16"
    count). One call covering one reference at a time is the expected shape;
    unparseable or unrequested arguments are ignored.
    """
    from scripture_fidelity.references import ReferenceError, parse_reference

    if not references:
        return 0.0
    parsed = {}
    for ref_str in references:
        try:
            parsed[ref_str] = parse_reference(ref_str)
        except ReferenceError:
            parsed[ref_str] = None

    covered: set[str] = set()
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            if call.function != tool_name:
                continue
            arg = (call.arguments or {}).get("reference", "")
            try:
                arg_parsed = parse_reference(str(arg))
            except ReferenceError:
                continue
            for ref_str in references:
                if parsed[ref_str] == arg_parsed:
                    covered.add(ref_str)
    return len(covered) / len(references)


def fail_fidelity(metrics: dict[str, float]) -> dict[str, float]:
    """Zero out fidelity metrics for a disobedient trial."""
    for key in _FIDELITY_KEYS:
        metrics[key] = 0.0
    metrics["cer"] = 1.0
    return metrics


def compute_multi_metrics(
    quotes: dict[str, str],
    truths: list[str],
    verses_per_ref: list[list[str]],
    references: list[str],
    language: str = "eng",
) -> dict[str, float]:
    """Score each requested reference against its own ground truth and
    return the mean of every metric across references. A reference with no
    quote contributes zeros (cer 1), so dropped passages visibly hurt."""
    per_ref = [
        compute_metrics(quotes.get(ref, ""), truth, verses, language)
        for ref, truth, verses in zip(references, truths, verses_per_ref)
    ]
    return {
        key: round(sum(m[key] for m in per_ref) / len(per_ref), 4)
        for key in per_ref[0]
    }


def quotation_fidelity():
    """Inspect scorer producing the full metric dict per sample.

    ``placeholder_ok`` is only meaningful for the output_buffer method
    (whether the model emitted a well-formed, correctly-referenced
    placeholder); ``tool_used`` only for tool_call/web_search (whether the
    model actually invoked its assigned tool). Each is fixed at 1.0 for
    methods it does not apply to. For multi-reference samples the metrics
    are per-reference means, and for tool_call ``tool_used`` is the
    fraction of requested references actually looked up via get_passage.
    A trial that disobeys its method's instructions (tool not used or only
    partially used, or placeholder malformed) fails: its fidelity metrics
    are zeroed so quoting from memory earns no credit.
    """
    from inspect_ai.scorer import Score, Target, mean, scorer, stderr
    from inspect_ai.solver import TaskState

    @scorer(
        metrics={key: [mean(), stderr()] for key in METRIC_KEYS},
        name="quotation_fidelity",
    )
    def _quotation_fidelity():
        async def score(state: TaskState, target: Target) -> Score:
            language = state.metadata.get("text_language", "eng")
            method = state.metadata.get("method")
            references = state.metadata.get("references")
            multi = bool(references) and state.metadata.get("set_size", 1) > 1

            if multi:
                quotes = extract_quotes(state.output.completion, references)
                metrics = compute_multi_metrics(
                    quotes,
                    state.metadata.get("ground_truth_texts", []),
                    state.metadata.get("ground_truth_verses_per_ref", []),
                    references,
                    language,
                )
                answer = "\n\n".join(
                    f'<quote ref="{ref}">{quotes.get(ref, "")}</quote>'
                    for ref in references
                )
            else:
                quote = extract_quote(state.output.completion)
                verses = state.metadata.get("ground_truth_verses", [])
                metrics = compute_metrics(quote, target.text, verses, language)
                answer = quote

            if method == "output_buffer":
                metrics["placeholder_ok"] = (
                    1.0 if state.store.get("placeholder_ok") else 0.0
                )
            else:
                metrics["placeholder_ok"] = 1.0
            expected_tool = METHOD_TOOLS.get(method)
            if expected_tool is None:
                metrics["tool_used"] = 1.0
            elif multi and method == "tool_call":
                metrics["tool_used"] = round(
                    tool_coverage(state.messages, expected_tool, references), 4
                )
            else:
                metrics["tool_used"] = (
                    1.0 if tool_was_used(state.messages, expected_tool) else 0.0
                )

            disobeyed = []
            if metrics["tool_used"] < 1.0:
                if metrics["tool_used"] == 0.0:
                    disobeyed.append(f"{expected_tool} tool not used")
                else:
                    disobeyed.append(
                        f"{expected_tool} covered only "
                        f"{metrics['tool_used']:.0%} of references"
                    )
            if metrics["placeholder_ok"] == 0.0:
                disobeyed.append("placeholder missing or malformed")
            if disobeyed:
                fail_fidelity(metrics)
                explanation = "FAILED (disobeyed prompt): " + "; ".join(disobeyed)
            else:
                explanation = (
                    f"similarity={metrics['similarity']:.3f} "
                    f"cer={metrics['cer']:.3f} exact={metrics['exact']:g}"
                )
            return Score(value=metrics, answer=answer, explanation=explanation)

        return score

    return _quotation_fidelity()
