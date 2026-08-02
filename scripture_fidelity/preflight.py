"""CLI-only dependency checks before a dry or real study run.

Validates that configured model providers, Bible translation APIs, and
(when used) the web-search provider accept a live probe request. Intended
for local ``scripture-fidelity run`` only — not the research API path.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from scripture_fidelity.bible.service import get_provider
from scripture_fidelity.config import ModelConfig, StudyConfig, TranslationConfig
from scripture_fidelity.references import parse_reference

# Keep the probe cheap: one short verse and a small search budget.
_PROBE_OBJECTIVE = (
    "The exact text of John 3:16 in the Berean Standard Bible translation"
)
_PROBE_QUERIES = ["John 3:16 Berean Standard Bible text"]
_PROBE_MAX_CHARS = 1000


@dataclass(frozen=True)
class CheckResult:
    """One dependency probe outcome."""

    category: str
    target: str
    ok: bool
    detail: str


def run_dependency_checks(config: StudyConfig) -> list[CheckResult]:
    """Probe every configured external dependency; never stop on first error."""
    return asyncio.run(_run_dependency_checks(config))


async def _run_dependency_checks(config: StudyConfig) -> list[CheckResult]:
    probe_ref = config.references[0].ref
    tasks: list[asyncio.Task[CheckResult]] = [
        asyncio.create_task(
            _check_model(model),
            name=f"model:{model.inspect_model}",
        )
        for model in config.models
    ]
    tasks.extend(
        asyncio.create_task(
            _check_translation(translation, probe_ref),
            name=f"translation:{translation.id}",
        )
        for translation in config.translations
    )
    if "web_search" in config.methods:
        tasks.append(
            asyncio.create_task(_check_web_search(), name="web_search:Parallel.ai")
        )

    return list(await asyncio.gather(*tasks))


async def _check_model(model: ModelConfig) -> CheckResult:
    target = model.inspect_model
    try:
        from inspect_ai.model import ChatMessageUser, get_model

        llm = get_model(model.inspect_model, **model.model_args)
        output = await llm.generate(
            [ChatMessageUser(content="Reply with the single word: ok")]
        )
        completion = (output.completion or "").strip()
        if not completion:
            return CheckResult(
                "model",
                target,
                False,
                "API responded but completion was empty",
            )
        preview = completion.replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:57] + "..."
        return CheckResult("model", target, True, f"ok ({preview!r})")
    except Exception as exc:  # noqa: BLE001 — collect all probe failures
        return CheckResult("model", target, False, _format_error(exc))


async def _check_translation(
    translation: TranslationConfig, ref_str: str
) -> CheckResult:
    target = f"{translation.id} ({translation.api})"
    try:
        provider = get_provider(translation.api)
        passage = await provider.get_passage(
            translation.api_bible_id,
            parse_reference(ref_str),
            translation.id,
        )
        if not passage.verses:
            return CheckResult(
                "translation",
                target,
                False,
                f"API returned no verses for {ref_str!r}",
            )
        if not passage.text.strip():
            return CheckResult(
                "translation",
                target,
                False,
                f"API returned empty text for {ref_str!r}",
            )
        return CheckResult(
            "translation",
            target,
            True,
            f"fetched {ref_str} ({len(passage.text)} chars, "
            f"{len(passage.verses)} verses)",
        )
    except Exception as exc:  # noqa: BLE001 — collect all probe failures
        return CheckResult("translation", target, False, _format_error(exc))


async def _check_web_search() -> CheckResult:
    target = "Parallel.ai"
    api_key = os.environ.get("PARALLEL_API_KEY", "")
    if not api_key:
        return CheckResult(
            "web_search",
            target,
            False,
            "PARALLEL_API_KEY is not set",
        )
    try:
        from parallel import AsyncParallel

        client = AsyncParallel(api_key=api_key)
        try:
            result = await client.search(
                objective=_PROBE_OBJECTIVE,
                search_queries=_PROBE_QUERIES,
                max_chars_total=_PROBE_MAX_CHARS,
            )
        finally:
            await client.close()
        results = getattr(result, "results", None) or []
        if not results:
            return CheckResult(
                "web_search",
                target,
                False,
                "API responded but returned no search results",
            )
        return CheckResult(
            "web_search",
            target,
            True,
            f"ok ({len(results)} results)",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("web_search", target, False, _format_error(exc))


def _format_error(exc: BaseException) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    message = message.replace("\n", " ")
    if len(message) > 200:
        message = message[:197] + "..."
    return f"{exc.__class__.__name__}: {message}"


def print_dependency_report(
    results: list[CheckResult], console: Console | None = None
) -> int:
    """Print a Rich table of probe results. Returns the failure count."""
    out = console or Console()
    table = Table(title="Dependency checks")
    table.add_column("Category")
    table.add_column("Target")
    table.add_column("Status")
    table.add_column("Detail")

    failures = 0
    for result in results:
        if result.ok:
            status = "[green]OK[/green]"
        else:
            status = "[red]FAIL[/red]"
            failures += 1
        table.add_row(result.category, result.target, status, result.detail)
    out.print(table)

    passed = len(results) - failures
    if failures:
        out.print(
            f"[red]Dependency checks:[/red] {passed} passed, "
            f"[bold]{failures} failed[/bold]"
        )
    else:
        out.print(
            f"[green]Dependency checks:[/green] {passed} passed, 0 failed"
        )
    return failures
