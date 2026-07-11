"""Grid expansion, ground-truth prefetch, and eval execution."""

from __future__ import annotations

import asyncio
import itertools
import json
from datetime import datetime
from pathlib import Path

from scripture_fidelity.bible.base import Passage
from scripture_fidelity.bible.service import PassageService
from scripture_fidelity.config import StudyConfig, TranslationConfig
from scripture_fidelity.references import parse_reference
from scripture_fidelity.task import build_task


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def raise_file_descriptor_limit() -> None:
    """Raise the soft open-file limit to the hard limit (POSIX only).

    Large grids open many eval log files at once; the macOS default soft
    limit of 256 is easily exceeded ("Too many open files").
    """
    try:
        import resource
    except ImportError:  # non-POSIX (e.g. Windows)
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except (ValueError, OSError):
            pass


async def prefetch_passages(
    config: StudyConfig, service: PassageService
) -> dict[str, dict[str, Passage]]:
    """Fetch ground truth for every translation x reference combination.

    Returns {translation_id: {ref_string: Passage}}. Raises on any failure
    so a run never starts with incomplete ground truth.
    """

    async def fetch(translation: TranslationConfig, ref_str: str):
        passage = await service.get(translation, parse_reference(ref_str))
        return translation.id, ref_str, passage

    tasks = [
        fetch(translation, ref.ref)
        for translation in config.translations
        for ref in config.references
    ]
    results: dict[str, dict[str, Passage]] = {}
    for translation_id, ref_str, passage in await asyncio.gather(*tasks):
        results.setdefault(translation_id, {})[ref_str] = passage
    return results


def build_tasks(
    config: StudyConfig,
    passages: dict[str, dict[str, Passage]],
    service: PassageService,
) -> list:
    """One Inspect task per (method, translation, language, temperature,
    set size)."""
    return [
        build_task(
            method=method,
            translation=translation,
            language=language,
            temperature=temperature,
            references=config.references,
            passages=passages[translation.id],
            service=service,
            set_size=set_size,
        )
        for method, translation, language, temperature, set_size in itertools.product(
            config.methods,
            config.translations,
            config.languages,
            config.temperatures,
            config.set_sizes,
        )
    ]


def run_study(
    config: StudyConfig,
    run_dir: Path,
    epochs: int = 1,
    max_connections: int = 10,
    max_tasks: int = 4,
    display: str = "rich",
    cache_dir: str | Path | None = None,
) -> Path:
    """Execute the full study grid; returns the Inspect log directory."""
    from inspect_ai import eval as inspect_eval

    raise_file_descriptor_limit()

    service = (
        PassageService(cache_dir) if cache_dir is not None else PassageService()
    )
    passages = asyncio.run(prefetch_passages(config, service))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    tasks = build_tasks(config, passages, service)
    models = [m.inspect_model for m in config.models]
    log_dir = run_dir / "logs"

    inspect_eval(
        tasks=tasks,
        model=models,
        epochs=epochs,
        log_dir=str(log_dir),
        max_connections=max_connections,
        max_tasks=max_tasks,
        display=display,
        # A single flaky sample (e.g. transient provider error) must not
        # abort a large grid: retry it, then record the error and move on.
        retry_on_error=3,
        fail_on_error=False,
        # Without these, Inspect retries transient HTTP errors *forever*
        # (backing off up to 30 min between attempts) and a hung connection
        # never times out — either can stall the run indefinitely on its
        # last task. Bound each attempt and the retry budget so a stuck
        # request becomes a sample error handled by retry_on_error above.
        attempt_timeout=300,
        timeout=1800,
        max_retries=5,
    )
    return log_dir
