"""Grid expansion, ground-truth prefetch, and eval execution."""

from __future__ import annotations

import asyncio
import itertools
import json
import shutil
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


def _prefetch(
    config: StudyConfig, cache_dir: str | Path | None
) -> tuple[dict[str, dict[str, Passage]], PassageService]:
    """Raise the fd limit and fetch all ground truth for the grid."""
    raise_file_descriptor_limit()
    service = (
        PassageService(cache_dir) if cache_dir is not None else PassageService()
    )
    passages = asyncio.run(prefetch_passages(config, service))
    return passages, service


def _run_eval(
    config: StudyConfig,
    passages: dict[str, dict[str, Passage]],
    service: PassageService,
    log_dir: str | Path,
    epochs: int,
    max_connections: int,
    max_tasks: int,
    display: str,
) -> None:
    """Execute the Inspect eval for the grid, writing logs to ``log_dir``.

    Inspect has no in-memory log mode; every caller must supply a writable
    ``log_dir`` (a run directory for the CLI, a temp dir for the API).
    """
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    tasks = build_tasks(config, passages, service)
    # Build model objects rather than passing bare strings so provider-specific
    # model args (e.g. Together's stream=true, required by some models) apply
    # per model without affecting the others.
    models = [
        get_model(m.inspect_model, **m.model_args) for m in config.models
    ]

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


def run_study(
    config: StudyConfig,
    run_dir: Path,
    epochs: int = 1,
    max_connections: int = 10,
    max_tasks: int = 4,
    display: str = "rich",
    cache_dir: str | Path | None = None,
) -> Path:
    """Execute the full study grid; returns the Inspect log directory.

    Persists the run to ``run_dir`` (config.json, logs/, export/). The API
    entry point (:func:`run_study_in_memory`) shares the eval core but writes
    nothing under ``run_dir``.
    """
    passages, service = _prefetch(config, cache_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    run_record = config.to_dict()
    run_record["call_accounting"] = call_accounting(config, epochs)
    run_record["generation_controls"] = GENERATION_CONTROLS
    (run_dir / "config.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log_dir = run_dir / "logs"
    _run_eval(
        config, passages, service, log_dir, epochs,
        max_connections, max_tasks, display,
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


def build_report(trial_rows: list[dict]) -> dict:
    """Recompute the report the CLI prints, as plain JSON-able data.

    Mirrors the CLI report: a per-permutation detail table plus mean metrics
    aggregated by each dimension that takes more than one value. Computed from
    the same trial rows the export writes, so the API's report matches the CLI.
    """
    from scripture_fidelity.report.data import (
        DIMENSIONS,
        REPORT_METRICS,
        aggregate,
        rows_from_trial_dicts,
    )

    rows = rows_from_trial_dicts(trial_rows)

    def _metrics(means: dict) -> dict:
        return {m: means[m] for m in REPORT_METRICS if m in means}

    detail_dims = tuple(
        d for d in DIMENSIONS if d != "language_match"
    )
    detail = [
        {
            "key": dict(zip(detail_dims, key)),
            "metrics": _metrics(means),
            "count": count,
        }
        for key, means, count in aggregate(rows, detail_dims)
    ]

    aggregates: dict[str, list] = {}
    for dim in DIMENSIONS:
        if len({getattr(r, dim) for r in rows}) > 1:
            aggregates[f"by_{dim}"] = [
                {"key": key[0], "metrics": _metrics(means), "count": count}
                for key, means, count in aggregate(rows, (dim,))
            ]

    return {
        "metrics": REPORT_METRICS,
        "detail": detail,
        "aggregates": aggregates,
    }


def run_study_in_memory(
    config: StudyConfig,
    epochs: int = 1,
    max_connections: int = 10,
    max_tasks: int = 4,
    cache_dir: str | Path | None = None,
) -> dict:
    """Execute the grid and return the full result package as plain data.

    Runs the same eval as :func:`run_study` but writes nothing to a persistent
    results directory: Inspect logs go to a temporary directory that is deleted
    before returning. The returned dict carries the same artifacts the CLI
    writes to ``export/`` (manifest, trials, source fixtures, method configs,
    scoring config) plus the recomputed report, so an API client receives
    exactly the data the command line produces.
    """
    import tempfile

    from scripture_fidelity.export import (
        build_method_configs,
        build_run_manifest,
        build_scoring_config,
        build_source_fixtures,
        build_trial_rows,
    )

    passages, service = _prefetch(config, cache_dir)

    run_id = new_run_id()
    tmp_dir = Path(tempfile.mkdtemp(prefix="sf-run-"))
    try:
        log_dir = tmp_dir / "logs"
        _run_eval(
            config, passages, service, log_dir, epochs,
            max_connections, max_tasks, display="none",
        )
        trial_rows = build_trial_rows(log_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = build_run_manifest(config, trial_rows, epochs, run_id)
    return {
        "run_id": run_id,
        "manifest": manifest,
        "trials": trial_rows,
        "source_fixtures": build_source_fixtures(config, passages),
        "method_configs": build_method_configs(config),
        "scoring_config": build_scoring_config(),
        "report": build_report(trial_rows),
    }
