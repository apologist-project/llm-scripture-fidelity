"""Method-specific Inspect solvers and tools."""

from __future__ import annotations

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


@solver
def output_buffer_transform() -> Solver:
    """Post-generation transform phase for the output_buffer method."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raw = state.output.completion
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


def solver_chain(
    method: str,
    language: str,
    translation: TranslationConfig,
    service: PassageService,
) -> list[Solver]:
    """Build the solver chain for one study method."""
    chain: list[Solver] = [system_message(system_prompt(language))]
    if method == "tool_call":
        chain.append(use_tools(get_passage(translation, service)))
    chain.append(generate())
    if method == "output_buffer":
        chain.append(output_buffer_transform())
    return chain
