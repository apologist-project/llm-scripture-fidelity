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


def analyze_final_output(completion: str) -> dict:
    """Structural analysis of the final user-visible completion under the
    declared renderer: <quote> blocks are the required wrapper markup and
    are removed by the renderer; any non-whitespace text outside them is
    extraneous. Missing tags yield zero blocks; nested or unbalanced tags
    leave residual markup that shows up as extraneous text or inside spans.
    """
    matches = list(QUOTE_ATTR_RE.finditer(completion or ""))
    outside = QUOTE_ATTR_RE.sub("", completion or "").strip()
    return {
        "quote_block_count": len(matches),
        "has_extraneous_text": bool(outside),
        "spans": [m.group(3).strip() for m in matches],
    }


def final_output_exact(completion: str, ground_truth: str) -> float:
    """1.0 only when the final completion is exactly one quote block whose
    rendered content matches the ground truth, with nothing else around it."""
    structure = analyze_final_output(completion)
    return (
        1.0
        if (
            structure["quote_block_count"] == 1
            and not structure["has_extraneous_text"]
            and collapse_ws(structure["spans"][0]) == collapse_ws(ground_truth)
        )
        else 0.0
    )


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
    "quote_span_exact",
    "final_output_exact",
    "extraneous_text",
    "quote_block_count",
    "method_adherence",
    "end_to_end_exact",
    "selection_correct",
    "lookup_ok",
    "replacement_ok",
]

# Tool each method is instructed to use (None = no tool expected)
METHOD_TOOLS = {"tool_call": "get_passage", "web_search": "search_web"}


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


def tool_coverage_by_reference(
    messages, tool_name: str, references: list[str]
) -> dict[str, float]:
    """Return an invocation-coverage flag for each requested reference."""
    from scripture_fidelity.references import ReferenceError, parse_reference

    parsed = {}
    for ref_str in references:
        try:
            parsed[ref_str] = parse_reference(ref_str)
        except ReferenceError:
            parsed[ref_str] = None
    covered = {ref: 0.0 for ref in references}
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            if call.function != tool_name:
                continue
            try:
                called = parse_reference(str((call.arguments or {}).get("reference", "")))
            except ReferenceError:
                continue
            for ref, expected in parsed.items():
                if expected == called:
                    covered[ref] = 1.0
    return covered


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

    Every component is reported independently — observed text fidelity is
    never overwritten when the model disobeys its method's instructions:

    - text fidelity (exact/normalized/similarity/cer/verse_coverage) is
      always the observed comparison of the extracted quote span(s);
    - ``quote_span_exact`` is the exactness of the extracted quote span;
    - ``final_output_exact`` additionally requires the whole final
      completion to be exactly the required quote block(s) with no
      extraneous text (the declared renderer removes only the wrapper
      markup — it never selects among multiple blocks);
    - ``extraneous_text`` / ``quote_block_count`` describe the structure of
      the final completion;
    - ``placeholder_ok`` (buffer_transform) and ``tool_used``
      (tool_call/web_search) report method compliance and are fixed at 1.0
      where not applicable; ``method_adherence`` is their conjunction;
    - ``end_to_end_exact`` = final_output_exact AND method_adherence.

    For multi-reference samples the text-fidelity metrics are per-reference
    means, and for tool_call ``tool_used`` is the fraction of requested
    references actually looked up via get_passage.
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
            completion = state.output.completion

            structure = analyze_final_output(completion)
            per_reference = []
            if multi:
                quotes = extract_quotes(completion, references)
                truths = state.metadata.get("ground_truth_texts", [])
                verses_per_ref = state.metadata.get(
                    "ground_truth_verses_per_ref", []
                )
                metrics = compute_multi_metrics(
                    quotes,
                    truths,
                    verses_per_ref,
                    references,
                    language,
                )
                spans_exact = metrics["exact"]
                output_exact = (
                    1.0
                    if (
                        structure["quote_block_count"] == len(references)
                        and not structure["has_extraneous_text"]
                        and all(
                            collapse_ws(quotes.get(ref, "")) == collapse_ws(truth)
                            for ref, truth in zip(references, truths)
                        )
                    )
                    else 0.0
                )
                answer = "\n\n".join(
                    f'<quote ref="{ref}">{quotes.get(ref, "")}</quote>'
                    for ref in references
                )
                per_reference = [
                    {
                        "reference": ref,
                        "answer": quotes.get(ref, ""),
                        "metrics": compute_metrics(
                            quotes.get(ref, ""), truth, verses, language
                        ),
                    }
                    for ref, truth, verses in zip(
                        references, truths, verses_per_ref
                    )
                ]
            else:
                quote = extract_quote(completion)
                verses = state.metadata.get("ground_truth_verses", [])
                metrics = compute_metrics(quote, target.text, verses, language)
                spans_exact = metrics["exact"]
                output_exact = final_output_exact(completion, target.text)
                answer = quote
                per_reference = [
                    {
                        "reference": state.metadata.get("reference"),
                        "answer": quote,
                        "metrics": dict(metrics),
                    }
                ]

            metrics["quote_span_exact"] = spans_exact
            metrics["final_output_exact"] = output_exact
            metrics["extraneous_text"] = (
                1.0 if structure["has_extraneous_text"] else 0.0
            )
            metrics["quote_block_count"] = float(structure["quote_block_count"])

            if method in ("buffer_transform", "buffer_transform_selection"):
                metrics["placeholder_ok"] = (
                    1.0 if state.store.get("placeholder_ok") else 0.0
                )
            else:
                metrics["placeholder_ok"] = 1.0
            if method == "buffer_transform_selection":
                metrics["selection_correct"] = (
                    1.0 if state.store.get("selection_correct") else 0.0
                )
                metrics["lookup_ok"] = 1.0 if state.store.get("lookup_ok") else 0.0
                metrics["replacement_ok"] = (
                    1.0 if state.store.get("replacement_ok") else 0.0
                )
            else:
                metrics["selection_correct"] = 1.0
                metrics["lookup_ok"] = 1.0
                metrics["replacement_ok"] = 1.0
            expected_tool = METHOD_TOOLS.get(method)
            if expected_tool is None:
                metrics["tool_used"] = 1.0
            elif multi and method == "tool_call":
                coverage_by_ref = tool_coverage_by_reference(
                    state.messages, expected_tool, references
                )
                metrics["tool_used"] = round(
                    sum(coverage_by_ref.values()) / len(references), 4
                )
                for item in per_reference:
                    item["metrics"]["tool_used"] = coverage_by_ref[
                        item["reference"]
                    ]
            else:
                metrics["tool_used"] = (
                    1.0 if tool_was_used(state.messages, expected_tool) else 0.0
                )

            metrics["method_adherence"] = (
                1.0
                if metrics["tool_used"] == 1.0 and metrics["placeholder_ok"] == 1.0
                else 0.0
            )
            metrics["end_to_end_exact"] = (
                1.0
                if metrics["final_output_exact"] == 1.0
                and metrics["method_adherence"] == 1.0
                and metrics["selection_correct"] == 1.0
                and metrics["lookup_ok"] == 1.0
                and metrics["replacement_ok"] == 1.0
                else 0.0
            )

            failure_tags = []
            if metrics["tool_used"] < 1.0:
                if metrics["tool_used"] == 0.0:
                    failure_tags.append(f"{expected_tool} tool not used")
                else:
                    failure_tags.append(
                        f"{expected_tool} covered only "
                        f"{metrics['tool_used']:.0%} of references"
                    )
            if metrics["placeholder_ok"] == 0.0:
                failure_tags.append("placeholder missing or malformed")
            if method == "buffer_transform_selection":
                if metrics["selection_correct"] == 0.0:
                    failure_tags.append("wrong or unparseable reference selected")
                if metrics["lookup_ok"] == 0.0:
                    failure_tags.append("selected reference lookup failed")
            explanation = (
                f"similarity={metrics['similarity']:.3f} "
                f"cer={metrics['cer']:.3f} exact={metrics['exact']:g} "
                f"end_to_end_exact={metrics['end_to_end_exact']:g}"
            )
            if failure_tags:
                explanation += (
                    " | method noncompliance: " + "; ".join(failure_tags)
                )
            return Score(
                value=metrics,
                answer=answer,
                explanation=explanation,
                metadata={
                    "raw_output": state.store.get("raw_output") or completion,
                    "final_output": completion,
                    "failure_tags": failure_tags,
                    "selected_reference_raw": state.store.get(
                        "selected_reference_raw"
                    ),
                    "selected_reference_parsed": state.store.get(
                        "selected_reference_parsed"
                    ),
                    "lookup_fixture_id": state.store.get("lookup_fixture_id"),
                    "per_reference": per_reference,
                },
            )

        return score

    return _quotation_fidelity()
