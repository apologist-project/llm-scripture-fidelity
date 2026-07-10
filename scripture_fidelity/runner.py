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
    """One Inspect task per (method, translation, language, temperature)."""
    return [
        build_task(
            method=method,
            translation=translation,
            language=language,
            temperature=temperature,
            references=config.references,
            passages=passages[translation.id],
            service=service,
        )
        for method, translation, language, temperature in itertools.product(
            config.methods,
            config.translations,
            config.languages,
            config.temperatures,
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
    )
    return log_dir
