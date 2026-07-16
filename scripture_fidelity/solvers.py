"""Method-specific Inspect solvers and tools."""

from __future__ import annotations

import os
import re

from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message, use_tools
from inspect_ai.tool import Tool, tool

from scripture_fidelity.bible.service import PassageService
from scripture_fidelity.config import TranslationConfig
from scripture_fidelity.prompts import system_prompt
from scripture_fidelity.references import ReferenceError, parse_reference

PLACEHOLDER_RE = re.compile(r"\{\{\s*QUOTE\s*:\s*([^{}]+?)\s*\}\}")


def apply_buffer_transform(
    text: str, expected_ref: str, ground_truth: str
) -> tuple[str, bool]:
    """Replace {{QUOTE:<ref>}} placeholders whose reference matches the
    expected one with the ground-truth text.

    Returns (transformed_text, placeholder_ok) where placeholder_ok is True
    only when exactly one placeholder was emitted and its reference resolves
    to the expected reference. Non-matching placeholders are left in place.
    """
    try:
        expected = parse_reference(expected_ref)
    except ReferenceError:
        return text, False

    matches = list(PLACEHOLDER_RE.finditer(text or ""))
    replaced = 0

    def _sub(m: re.Match) -> str:
        nonlocal replaced
        try:
            if parse_reference(m.group(1)) == expected:
                replaced += 1
                return ground_truth
        except ReferenceError:
            pass
        return m.group(0)

    transformed = PLACEHOLDER_RE.sub(_sub, text or "")
    ok = len(matches) == 1 and replaced == 1
    return transformed, ok


def apply_buffer_transform_multi(
    text: str, expected: list[tuple[str, str]]
) -> tuple[str, bool]:
    """Replace {{QUOTE:<ref>}} placeholders for several expected references.

    ``expected`` is a list of (reference, ground truth text) pairs. Returns
    (transformed_text, placeholder_ok) where placeholder_ok is True only
    when every expected reference was matched by exactly one placeholder
    and no extra placeholders were emitted.
    """
    targets = []
    for ref_str, truth in expected:
        try:
            targets.append((parse_reference(ref_str), truth))
        except ReferenceError:
            return text, False

    matches = list(PLACEHOLDER_RE.finditer(text or ""))
    replaced: dict[int, int] = {}

    def _sub(m: re.Match) -> str:
        try:
            parsed = parse_reference(m.group(1))
        except ReferenceError:
            return m.group(0)
        for i, (ref, truth) in enumerate(targets):
            if parsed == ref:
                replaced[i] = replaced.get(i, 0) + 1
                return truth
        return m.group(0)

    transformed = PLACEHOLDER_RE.sub(_sub, text or "")
    ok = len(matches) == len(targets) and all(
        replaced.get(i) == 1 for i in range(len(targets))
    )
    return transformed, ok


@solver
def buffer_transform_solver(multi: bool = False) -> Solver:
    """Post-generation transform phase for the buffer_transform method."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raw = state.output.completion
        if multi:
            expected = list(
                zip(
                    state.metadata["references"],
                    state.metadata["ground_truth_texts"],
                )
            )
            transformed, ok = apply_buffer_transform_multi(raw, expected)
        else:
            transformed, ok = apply_buffer_transform(
                raw, state.metadata["reference"], state.target.text
            )
        state.output.completion = transformed
        state.store.set("raw_output", raw)
        state.store.set("placeholder_ok", ok)
        return state

    return solve


async def apply_buffer_transform_selection(
    text: str, expected_ref: str, translation: TranslationConfig, lookup
) -> tuple[str, dict]:
    """Apply the selection-scenario transform to a completion.

    The model must have emitted exactly one {{QUOTE:<reference>}}
    placeholder naming the reference it selected. The placeholder is looked
    up via ``lookup(translation, parsed_reference)`` using the *selected*
    reference — never ``expected_ref`` — so a wrong but valid selection is
    replaced with the wrong passage's text, not the expected scenario text.

    Returns (transformed_text, result) where result records the raw
    selection, parse result, lookup fixture id, and each component outcome.
    """
    matches = list(PLACEHOLDER_RE.finditer(text or ""))
    selected_raw = matches[0].group(1) if matches else ""

    parsed = None
    if selected_raw:
        try:
            parsed = parse_reference(selected_raw)
        except ReferenceError:
            parsed = None

    transformed = text
    lookup_ok = False
    lookup_fixture = ""
    replaced = False
    if len(matches) == 1 and parsed is not None:
        lookup_fixture = f"{translation.source_key}:{parsed.usfm()}"
        try:
            passage = await lookup(translation, parsed)
        except Exception:
            passage = None
        if passage is not None:
            lookup_ok = True
            m = matches[0]
            transformed = text[: m.start()] + passage.text + text[m.end() :]
            replaced = True

    selection_correct = False
    if parsed is not None:
        try:
            selection_correct = parsed == parse_reference(expected_ref)
        except ReferenceError:
            selection_correct = False

    result = {
        "selected_reference_raw": selected_raw,
        "selected_reference_parsed": parsed.usfm() if parsed else "",
        "placeholder_count": len(matches),
        "placeholder_ok": len(matches) == 1 and parsed is not None,
        "selection_correct": selection_correct,
        "lookup_ok": lookup_ok,
        "lookup_fixture_id": lookup_fixture,
        "replacement_ok": replaced,
    }
    return transformed, result


@solver
def buffer_transform_selection_solver(
    translation: TranslationConfig, service: PassageService
) -> Solver:
    """Post-generation transform for the buffer_transform_selection method.

    See apply_buffer_transform_selection: replacement uses the model's own
    selected reference, and every component outcome is recorded in the
    store."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raw = state.output.completion
        transformed, result = await apply_buffer_transform_selection(
            raw, state.metadata["reference"], translation, service.get
        )
        state.output.completion = transformed
        state.store.set("raw_output", raw)
        for key, value in result.items():
            state.store.set(key, value)
        return state

    return solve


@tool
def get_passage(translation: TranslationConfig, service: PassageService) -> Tool:
    async def execute(reference: str) -> str:
        """Get the exact text of a Bible passage in the study's translation.

        Args:
            reference: Scripture reference, e.g. "John 3:16",
                "Romans 8:38-39", or "Psalm 117".

        Returns:
            The exact passage text.
        """
        try:
            passage = await service.get_by_ref_string(translation, reference)
        except ReferenceError as e:
            return f"Error: {e}"
        return passage.text

    return execute


@tool
def search_web() -> Tool:
    async def execute(objective: str, search_queries: list[str]) -> str:
        """Search the web for the exact text of a Bible passage.

        Args:
            objective: What you are trying to find, e.g. "The exact text of
                John 3:16 in the Berean Standard Bible translation".
            search_queries: One to five web search queries, e.g.
                ["John 3:16 Berean Standard Bible text"].

        Returns:
            Search results as titled excerpts from web pages.
        """
        from parallel import AsyncParallel

        api_key = os.environ.get("PARALLEL_API_KEY", "")
        if not api_key:
            return "Error: PARALLEL_API_KEY is not set"
        client = AsyncParallel(api_key=api_key)
        try:
            result = await client.search(
                objective=objective,
                search_queries=search_queries[:5],
                max_chars_total=6000,
            )
        finally:
            await client.close()
        if not result.results:
            return "No results found."
        parts = []
        for r in result.results:
            excerpts = "\n".join(r.excerpts or [])
            parts.append(f"[{r.title}]({r.url})\n{excerpts}")
        return "\n\n---\n\n".join(parts)

    return execute


def solver_chain(
    method: str,
    language: str,
    translation: TranslationConfig,
    service: PassageService,
    multi: bool = False,
) -> list[Solver]:
    """Build the solver chain for one study method. ``multi`` selects the
    multi-reference system prompt and placeholder transform."""
    chain: list[Solver] = [system_message(system_prompt(language, multi=multi))]
    if method == "tool_call":
        chain.append(use_tools(get_passage(translation, service)))
    elif method == "web_search":
        chain.append(use_tools(search_web()))
    chain.append(generate())
    if method == "buffer_transform":
        chain.append(buffer_transform_solver(multi=multi))
    elif method == "buffer_transform_selection":
        chain.append(buffer_transform_selection_solver(translation, service))
    return chain
