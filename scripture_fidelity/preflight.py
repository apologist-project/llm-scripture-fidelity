"""CLI-only dependency checks before a dry or real study run.

Validates that configured model providers, Bible translation APIs, and
(when used) the web-search provider accept a live probe request shaped like
the study grid. Intended for local ``scripture-fidelity run`` only — not the
research API path.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from scripture_fidelity.bible.service import get_provider
from scripture_fidelity.config import ModelConfig, StudyConfig, TranslationConfig
from scripture_fidelity.references import parse_reference

_PROBE_MAX_CHARS = 1000
_ACTIVE_PROBE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sf_preflight_probe", default=None
)
_REQUEST_JSON_RE = re.compile(
    r"Request:\s*\{.*?\}\s*(?=(?:BadRequestError|APIStatusError|Error|Exception|\Z))",
    re.DOTALL,
)
# Prefer double-quoted API messages (may contain single quotes like 'temperature').
_API_MESSAGE_DOUBLE_RE = re.compile(
    r"[\"']message[\"']\s*:\s*\"([^\"]+)\""
)
_API_MESSAGE_SINGLE_RE = re.compile(
    r"[\"']message[\"']\s*:\s*'([^']+)'"
)


@dataclass(frozen=True)
class CheckResult:
    """One dependency probe outcome."""

    category: str
    target: str
    ok: bool
    detail: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


def highest_temperature(config: StudyConfig) -> float | None:
    """Highest numeric temperature in the study grid, or None if all null."""
    numeric = [t for t in config.temperatures if t is not None]
    return max(numeric) if numeric else None


def probe_temperature_for_model(
    model: ModelConfig, config: StudyConfig
) -> float | None:
    """Temperature to send on a model probe (None omits the parameter)."""
    if not model.supports_temperature:
        return None
    return highest_temperature(config)


def model_probe_prompt(ref: str) -> str:
    return (
        f"Quote {ref} word for word in a single short reply. "
        "If you cannot, reply with the single word: ok"
    )


def _quiet_native_runtime_logs() -> None:
    """Suppress native gRPC/Abseil INFO noise before provider clients start."""
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GRPC_TRACE", "")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")


def run_dependency_checks(config: StudyConfig) -> list[CheckResult]:
    """Probe every configured external dependency; never stop on first error."""
    _quiet_native_runtime_logs()
    return asyncio.run(_run_dependency_checks(config))


async def _run_dependency_checks(config: StudyConfig) -> list[CheckResult]:
    probe_ref = config.references[0].ref
    first_translation = config.translations[0]
    tasks: list[asyncio.Task[CheckResult]] = [
        asyncio.create_task(
            _check_model(model, config),
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
            asyncio.create_task(
                _check_web_search(probe_ref, first_translation),
                name="web_search:Parallel.ai",
            )
        )

    return list(await asyncio.gather(*tasks))


async def _check_model(model: ModelConfig, config: StudyConfig) -> CheckResult:
    target = model.inspect_model
    probe_ref = config.references[0].ref
    temperature = probe_temperature_for_model(model, config)
    temp_note = (
        "provider default"
        if temperature is None
        else f"temperature={temperature:g}"
    )
    try:
        from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

        llm = get_model(model.inspect_model, **model.model_args)
        gen_config = (
            GenerateConfig()
            if temperature is None
            else GenerateConfig(temperature=temperature)
        )
        with _capture_probe_noise(target) as captured:
            output = await llm.generate(
                [ChatMessageUser(content=model_probe_prompt(probe_ref))],
                config=gen_config,
            )
        warn_msgs = tuple(captured)
        completion = (output.completion or "").strip()
        if not completion:
            return CheckResult(
                "model",
                target,
                False,
                f"API responded but completion was empty ({temp_note})",
                warn_msgs,
            )
        preview = completion.replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:57] + "..."
        return CheckResult(
            "model",
            target,
            True,
            f"ok ({temp_note}, {probe_ref!r} → {preview!r})",
            warn_msgs,
        )
    except Exception as exc:  # noqa: BLE001 — collect all probe failures
        detail = _format_error(exc)
        if _looks_like_temperature_rejection(detail):
            detail = (
                f"{detail} — mark supports_temperature=false for this model"
            )
        return CheckResult("model", target, False, detail)


async def _check_translation(
    translation: TranslationConfig, ref_str: str
) -> CheckResult:
    target = f"{translation.id} ({translation.api})"
    try:
        with _capture_probe_noise(target, model_scoped=False) as captured:
            provider = get_provider(translation.api)
            passage = await provider.get_passage(
                translation.api_bible_id,
                parse_reference(ref_str),
                translation.id,
            )
        warn_msgs = tuple(captured)
        if not passage.verses:
            return CheckResult(
                "translation",
                target,
                False,
                f"API returned no verses for {ref_str!r}",
                warn_msgs,
            )
        if not passage.text.strip():
            return CheckResult(
                "translation",
                target,
                False,
                f"API returned empty text for {ref_str!r}",
                warn_msgs,
            )
        return CheckResult(
            "translation",
            target,
            True,
            f"fetched {ref_str} ({len(passage.text)} chars, "
            f"{len(passage.verses)} verses)",
            warn_msgs,
        )
    except Exception as exc:  # noqa: BLE001 — collect all probe failures
        return CheckResult("translation", target, False, _format_error(exc))


async def _check_web_search(
    ref_str: str, translation: TranslationConfig
) -> CheckResult:
    target = "Parallel.ai"
    translation_label = translation.display_name or translation.id
    objective = (
        f"The exact text of {ref_str} in the {translation_label} translation"
    )
    queries = [f"{ref_str} {translation_label} text"]
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
            with _capture_probe_noise(target, model_scoped=False) as captured:
                result = await client.search(
                    objective=objective,
                    search_queries=queries,
                    max_chars_total=_PROBE_MAX_CHARS,
                )
        finally:
            await client.close()
        warn_msgs = tuple(captured)
        results = getattr(result, "results", None) or []
        if not results:
            return CheckResult(
                "web_search",
                target,
                False,
                "API responded but returned no search results",
                warn_msgs,
            )
        return CheckResult(
            "web_search",
            target,
            True,
            f"ok ({len(results)} results; {ref_str} / {translation.id})",
            warn_msgs,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("web_search", target, False, _format_error(exc))


@contextmanager
def _capture_probe_noise(target: str, *, model_scoped: bool = True):
    """Collect noise for one probe without leaking across concurrent probes.

    A ContextVar tags the active probe so root-logger handlers installed by
    sibling tasks ignore each other's records. Model probes only keep messages
    that mention that model; non-model probes ignore Inspect model-provider
    logs entirely.
    """
    captured: list[str] = []
    probe_id = f"{target}:{id(captured)}"

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno < logging.WARNING:
                return
            if _ACTIVE_PROBE.get() != probe_id:
                return
            msg = record.getMessage().strip()
            if not msg:
                return
            text = f"{record.name}: {msg}" if record.name else msg
            if not _message_belongs_to_probe(
                text, target, model_scoped=model_scoped, logger_name=record.name
            ):
                return
            captured.append(_truncate(text.replace("\n", " "), 200))

    handler = _Handler()
    handler.setLevel(logging.WARNING)
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    if prev_level > logging.WARNING or prev_level == logging.NOTSET:
        root.setLevel(logging.WARNING)
    token = _ACTIVE_PROBE.set(probe_id)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield captured
            for w in caught:
                text = str(w.message).strip()
                if not text:
                    continue
                labeled = f"{w.category.__name__}: {text}"
                if not _message_belongs_to_probe(
                    labeled, target, model_scoped=model_scoped
                ):
                    continue
                captured.append(_truncate(labeled.replace("\n", " "), 200))
    finally:
        _ACTIVE_PROBE.reset(token)
        root.removeHandler(handler)
        root.setLevel(prev_level)


def _message_belongs_to_probe(
    message: str,
    target: str,
    *,
    model_scoped: bool,
    logger_name: str = "",
) -> bool:
    if model_scoped:
        if target in message:
            return True
        bare = target.split("/", 1)[-1]
        # Inspect warnings typically say: model 'claude-opus-5'
        if re.search(rf"model\s+'{re.escape(bare)}'", message):
            return True
        if re.search(rf'model\s+"{re.escape(bare)}"', message):
            return True
        if re.search(rf"\bfor model {re.escape(bare)}\b", message):
            return True
        # Avoid treating generic ids like "model" as a substring match.
        if bare not in {"model", "latest"} and re.search(
            rf"(?<![A-Za-z0-9_./-]){re.escape(bare)}(?![A-Za-z0-9_./-])",
            message,
        ):
            return True
        return False
    # Translation / web-search probes: ignore model-provider chatter.
    if logger_name.startswith("inspect_ai.model"):
        return False
    if "inspect_ai.model" in message or "does not support the 'temperature'" in message:
        return False
    return True


def _truncate(message: str, limit: int) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _looks_like_temperature_rejection(detail: str) -> bool:
    lowered = detail.casefold()
    return "temperature" in lowered and (
        "unsupported parameter" in lowered
        or "not supported" in lowered
        or "does not support" in lowered
    )


def _format_error(exc: BaseException) -> str:
    """Prefer the underlying API reason over Inspect's request dump."""
    candidates: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text:
            candidates.append(text)
        current = current.__cause__ or current.__context__

    # Walk from the innermost cause outward for the most specific API message.
    for text in reversed(candidates):
        cleaned = _REQUEST_JSON_RE.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" :")
        match = _API_MESSAGE_DOUBLE_RE.search(cleaned) or _API_MESSAGE_SINGLE_RE.search(
            cleaned
        )
        if match:
            return _truncate(f"{type(exc).__name__}: {match.group(1)}", 200)
        if "Unsupported parameter" in cleaned or "Error code:" in cleaned:
            nested = re.search(
                r"BadRequestError\((['\"])(.*?)\1\)", cleaned, re.DOTALL
            )
            if nested:
                return _truncate(f"{type(exc).__name__}: {nested.group(2)}", 200)
            return _truncate(f"{type(exc).__name__}: {cleaned}", 200)

    message = candidates[0] if candidates else type(exc).__name__
    message = _REQUEST_JSON_RE.sub(" ", message)
    message = re.sub(r"\s+", " ", message).strip()
    return _truncate(f"{type(exc).__name__}: {message}", 200)


def print_dependency_report(
    results: list[CheckResult], console: Console | None = None
) -> int:
    """Print a Rich table of probe results. Returns the hard-failure count."""
    out = console or Console()
    table = Table(title="Dependency checks (live probes)")
    table.add_column("Category")
    table.add_column("Target")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_column("Warnings")

    failures = 0
    warns = 0
    for result in results:
        if not result.ok:
            status = "[red]FAIL[/red]"
            failures += 1
        elif result.has_warnings:
            status = "[yellow]WARN[/yellow]"
            warns += 1
        else:
            status = "[green]OK[/green]"
        warning_text = "; ".join(result.warnings) if result.warnings else ""
        table.add_row(
            result.category, result.target, status, result.detail, warning_text
        )
    out.print(table)

    passed = len(results) - failures
    summary = (
        f"Dependency checks: {passed} passed ({warns} with warnings), "
        f"{failures} failed"
    )
    if failures:
        out.print(f"[red]{summary}[/red]")
    elif warns:
        out.print(f"[yellow]{summary}[/yellow]")
        out.print(
            "[yellow]Warnings do not abort the run. Mark models that warn "
            "about temperature with supports_temperature=false, then "
            "re-check.[/yellow]"
        )
    else:
        out.print(f"[green]{summary}[/green]")
    return failures
