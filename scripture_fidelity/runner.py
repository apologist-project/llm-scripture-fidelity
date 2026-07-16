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


# Retry policy applied to every eval (see run_study) — surfaced in dry-run
# call accounting so the upper bound on provider calls is visible up front.
RETRY_ON_ERROR = 3
MAX_HTTP_RETRIES = 5

# Generation/sampling controls applied to every eval. Only temperature is
# varied by the study grid; everything else is left at the provider default
# and recorded as such so provenance is explicit (AP-08).
GENERATION_CONTROLS = {
    "temperature": "per-variant (see TEMPERATURES)",
    "top_p": "provider default",
    "top_k": "provider default",
    "max_tokens": "provider default",
    "reasoning_effort": "provider default",
    "reasoning_tokens": "provider default",
    "attempt_timeout": 300,
    "timeout": 1800,
    "max_retries": MAX_HTTP_RETRIES,
    "retry_on_error": RETRY_ON_ERROR,
}

# Planned generation attempts above this require explicit authorization
# (--confirm-large-run) before any provider call is made.
CALL_VOLUME_THRESHOLD = 10_000


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def call_accounting(config: StudyConfig, epochs: int) -> dict[str, int]:
    """Planned call volumes for a run: expected requests, per-reference
    observations, epochs, and retry upper bounds."""
    samples_per_epoch = config.permutation_count()
    planned_requests = samples_per_epoch * epochs
    observations_per_reference = (
        len(config.methods)
        * len(config.variant_pairs())
        * len(config.models)
        * len(config.temperatures)
        * len(config.set_sizes)
        * epochs
    )
    max_generation_attempts = planned_requests * (1 + RETRY_ON_ERROR)
    return {
        "language_pairs": len(config.variant_pairs()),
        "samples_per_epoch": samples_per_epoch,
        "epochs": epochs,
        "planned_requests": planned_requests,
        "observations_per_reference": observations_per_reference,
        "retry_on_error": RETRY_ON_ERROR,
        "max_http_retries_per_attempt": MAX_HTTP_RETRIES,
        "max_generation_attempts": max_generation_attempts,
    }


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

    Returns {translation_source_key: {ref_string: Passage}} keyed by the
    stable composite source identity so translations with overlapping
    display names or ids from different providers can never collide.
    Raises on any failure so a run never starts with incomplete ground
    truth. Only translations that appear in the study's variant pairs are
    fetched.
    """

    async def fetch(translation: TranslationConfig, ref_str: str):
        passage = await service.get(translation, parse_reference(ref_str))
        return translation.source_key, ref_str, passage

    translations = {t.source_key: t for _, t in config.variant_pairs()}
    tasks = [
        fetch(translation, ref.ref)
        for translation in translations.values()
        for ref in config.references
    ]
    results: dict[str, dict[str, Passage]] = {}
    for source_key, ref_str, passage in await asyncio.gather(*tasks):
        results.setdefault(source_key, {})[ref_str] = passage
    return results


def build_tasks(
    config: StudyConfig,
    passages: dict[str, dict[str, Passage]],
    service: PassageService,
) -> list:
    """One Inspect task per (method, (language, translation) pair,
    temperature, set size). Pairs come from the declared pairing mode, so
    crossed prompts are only generated when explicitly configured."""
    return [
        build_task(
            method=method,
            translation=translation,
            language=language,
            temperature=temperature,
            references=config.references,
            passages=passages[translation.source_key],
            service=service,
            set_size=set_size,
            pairing_mode=config.language_pairing_mode,
            protocol_role=config.protocol_role,
        )
        for method, (language, translation), temperature, set_size in itertools.product(
            config.methods,
            config.variant_pairs(),
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
    run_record = config.to_dict()
    run_record["call_accounting"] = call_accounting(config, epochs)
    run_record["generation_controls"] = GENERATION_CONTROLS
    (run_dir / "config.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False), encoding="utf-8"
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
        retry_on_error=RETRY_ON_ERROR,
        fail_on_error=False,
        # Without these, Inspect retries transient HTTP errors *forever*
        # (backing off up to 30 min between attempts) and a hung connection
        # never times out — either can stall the run indefinitely on its
        # last task. Bound each attempt and the retry budget so a stuck
        # request becomes a sample error handled by retry_on_error above.
        attempt_timeout=300,
        timeout=1800,
        max_retries=MAX_HTTP_RETRIES,
    )

    from scripture_fidelity.export import export_package

    export_package(
        config,
        passages,
        log_dir,
        run_dir / "export",
        epochs=epochs,
        run_id=run_dir.name,
    )
    return log_dir
