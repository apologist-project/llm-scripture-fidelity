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


def apply_output_buffer(
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


def apply_output_buffer_multi(
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
def output_buffer_transform(multi: bool = False) -> Solver:
    """Post-generation transform phase for the output_buffer method."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raw = state.output.completion
        if multi:
            expected = list(
                zip(
                    state.metadata["references"],
                    state.metadata["ground_truth_texts"],
                )
            )
            transformed, ok = apply_output_buffer_multi(raw, expected)
        else:
            transformed, ok = apply_output_buffer(
                raw, state.metadata["reference"], state.target.text
            )
        state.output.completion = transformed
        state.store.set("raw_output", raw)
        state.store.set("placeholder_ok", ok)
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
    if method == "output_buffer":
        chain.append(output_buffer_transform(multi=multi))
    return chain
