"""Auditable result package export.

Writes a normalized, self-contained package next to the Inspect logs:

- ``run-manifest.json``   run identity, protocol role, call accounting,
                          model provenance, and row-count reconciliation
- ``trials.jsonl``        one row per requested reference (multi-reference
                          prompts share a ``request_id``)
- ``source-fixtures.jsonl`` one row per (translation, reference) fixture
                          with provenance, hashes, rights, verification
- ``method-configs.json`` per-method solver/tool/transform declarations
- ``scoring-config.json`` metric definitions and normalization declaration

Reports can be recomputed from this package alone (see
``scripture_fidelity.report.data.rows_from_export``).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from scripture_fidelity.bible.base import TEXT_NORMALIZATION_VERSION, Passage
from scripture_fidelity.config import StudyConfig, fixture_id
from scripture_fidelity.scoring import METHOD_TOOLS, METRIC_KEYS

EXPORT_SCHEMA_VERSION = "2"


class ExportError(RuntimeError):
    """Raised when a result package cannot be exported safely."""


def _sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _structured_error(error) -> dict | None:
    if error is None:
        return None
    raw = " ".join(str(error).split())
    first = raw.split(" traceback=", 1)[0]
    class_match = re.search(
        r"(?:^|\b)([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))", first
    )
    error_class = class_match.group(1) if class_match else type(error).__name__
    if error_class == "str" and "timeout" in first.casefold():
        error_class = "TimeoutError"
    return {
        "error_class": error_class,
        "message": "Trial failed; consult restricted execution logs",
    }


def build_source_fixtures(
    config: StudyConfig, passages: dict[str, dict[str, Passage]]
) -> list[dict]:
    """One provenance row per (translation, reference) fixture.

    ``passages`` maps translation.source_key -> {ref_string: Passage}.
    Restricted sources are exported without text (hash and verification
    mode only); a restricted source with no declared verification mode
    blocks the export.
    """
    by_source_key = {t.source_key: t for t in config.translations}
    rows = []
    for source_key, by_ref in passages.items():
        translation = by_source_key[source_key]
        restricted = translation.rights == "restricted"
        if restricted and not translation.verification:
            raise ExportError(
                f"Translation {translation.id!r} is rights-restricted but has "
                "no verification mode; refusing to export its fixtures"
            )
        for ref_str, passage in by_ref.items():
            row = {
                "fixture_id": fixture_id(translation, ref_str),
                "source_api": translation.api,
                "source_bible_id": translation.api_bible_id,
                "translation_id": translation.id,
                "translation_name": translation.display_name,
                "language": translation.language,
                "reference": ref_str,
                "canonical_reference": fixture_id(translation, ref_str).rsplit(
                    ":", 1
                )[-1],
                "retrieved_at": passage.retrieved_at,
                "text_sha256": passage.text_sha256,
                "normalized_text_sha256": passage.text_sha256,
                "normalization_version": TEXT_NORMALIZATION_VERSION,
                "rights": translation.rights,
                "verification": translation.verification,
                "edition": translation.edition or None,
                "license_basis": translation.license_basis or None,
                "public_release": translation.public_release,
            }
            if not restricted:
                row["text"] = passage.text
                row["verses"] = [v.text for v in passage.verses]
            rows.append(row)
    return rows


def build_method_configs(config: StudyConfig) -> dict:
    """Declared solver/tool/transform shape for each configured method."""
    methods = {}
    for method in config.methods:
        methods[method] = {
            "tool": METHOD_TOOLS.get(method),
            "transform": (
                "buffer_transform_solver"
                if method == "buffer_transform"
                else "buffer_transform_selection_solver"
                if method == "buffer_transform_selection"
                else None
            ),
        }
    return methods


def build_scoring_config() -> dict:
    return {
        "metrics": METRIC_KEYS,
        "method_tools": METHOD_TOOLS,
        "final_output_renderer": (
            "quote-block wrapper markup (<quote>/<quote ref=...>) is required "
            "and removed by the renderer; any other text is extraneous"
        ),
        "text_normalization_version": TEXT_NORMALIZATION_VERSION,
        "end_to_end_exact": (
            "final_output_exact AND method_adherence AND selection_correct "
            "AND lookup_ok AND replacement_ok"
        ),
    }


def _sample_trial_rows(log, sample, requested_model: str) -> list[dict]:
    """Expand one Inspect sample into per-reference trial rows."""
    md = sample.metadata or {}
    scores = sample.scores or {}
    score = next(
        (s for s in scores.values() if isinstance(s.value, dict)), None
    )
    metrics = (
        {k: float(v) for k, v in score.value.items()} if score is not None else {}
    )
    score_md = (score.metadata or {}) if score is not None else {}
    resolved_model = str(getattr(sample.output, "model", "") or "") or None
    usage = getattr(sample.output, "usage", None)
    error = _structured_error(getattr(sample, "error", None))

    # run_id is shared across every task in one eval() invocation, so the
    # task identity (variant + model) must be part of the request id.
    execution_request_id = (
        f"{log.eval.run_id}:{log.eval.task}:{requested_model}:"
        f"{sample.id}:{sample.epoch}"
    )
    request_id = md.get("caller_request_id") or execution_request_id
    raw_output = score_md.get("raw_output")
    final_output = score_md.get("final_output")
    common = {
        "request_id": request_id,
        "execution_request_id": execution_request_id,
        "scenario_id": md.get("scenario_id"),
        "protocol_version": md.get("protocol_version"),
        "repetition": md.get("repetition", sample.epoch),
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "method": md.get("method"),
        "prompt_family": md.get("prompt_family", "method_specific"),
        "translation": md.get("translation"),
        "prompt_language": md.get("prompt_language"),
        "language_match": md.get("language_match"),
        "language_pairing_mode": md.get("language_pairing_mode"),
        "protocol_role": md.get("protocol_role"),
        "temperature": md.get("temperature"),
        "set_size": md.get("set_size", 1),
        "epoch": sample.epoch,
        "error": error,
        "retries": getattr(sample, "retries", None),
        "total_time": getattr(sample, "total_time", None),
        "working_time": getattr(sample, "working_time", None),
        "usage": usage.model_dump() if usage is not None else None,
        "prompt_source": md.get("prompt_source", "generated"),
        "prompt_sha256": md.get("prompt_sha256"),
        "effective_user_input_sha256": md.get("effective_user_input_sha256"),
        "source_fixture_id_requested": md.get("source_fixture_id_requested"),
        "source_document_supplied": md.get("source_document_supplied", False),
        "source_document_sha256": md.get("source_document_sha256"),
        "raw_output": raw_output,
        "raw_output_sha256": _sha256(raw_output),
        "final_output": final_output,
        "final_output_sha256": _sha256(final_output),
        "failure_tags": score_md.get("failure_tags", []),
        "selected_reference_raw": score_md.get("selected_reference_raw"),
        "selected_reference_parsed": score_md.get("selected_reference_parsed"),
        "lookup_fixture_id": score_md.get("lookup_fixture_id"),
        "request_metrics": metrics,
    }
    return _expand_references(md, metrics, score, score_md, common)


def _expand_references(
    md: dict, metrics: dict, score, score_md: dict, common: dict
) -> list[dict]:
    """One trial row per requested reference; a multi-reference request
    yields several rows sharing the request_id."""
    references = md.get("references")
    if references and md.get("set_size", 1) > 1:
        fixture_ids = md.get("fixture_ids") or [None] * len(references)
        per_reference = {
            str(item.get("reference")): item
            for item in score_md.get("per_reference", [])
            if isinstance(item, dict)
        }
        rows = []
        for i, (ref, fid) in enumerate(zip(references, fixture_ids)):
            observed = per_reference.get(str(ref), {})
            rows.append(
                {
                    **common,
                    "trial_id": f"{common['request_id']}:{i}",
                    "reference": ref,
                    "reference_index": i,
                    "fixture_id": fid,
                    "metrics": observed.get("metrics", {}),
                    "answer": observed.get("answer", ""),
                }
            )
        return rows
    return [
        {
            **common,
            "trial_id": f"{common['request_id']}:0",
            "reference": md.get("reference"),
            "reference_index": 0,
            "fixture_id": md.get("fixture_id"),
            "ref_type": md.get("ref_type"),
            "metrics": metrics,
            "answer": (score.answer or "") if score is not None else "",
        }
    ]


def build_trial_rows(log_dir: str | Path) -> list[dict]:
    """Flatten all Inspect logs into per-reference trial rows."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    rows: list[dict] = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        requested_model = str(log.eval.model)
        for sample in log.samples or []:
            rows.extend(_sample_trial_rows(log, sample, requested_model))
    return rows


def build_run_manifest(
    config: StudyConfig,
    trial_rows: list[dict],
    epochs: int,
    run_id: str = "",
    started_at: str | None = None,
    ended_at: str | None = None,
) -> dict:
    from scripture_fidelity.runner import GENERATION_CONTROLS, call_accounting
    from scripture_fidelity.provenance import build_identity, execution_environment

    completed = sum(1 for r in trial_rows if not r["error"])
    errored = sum(1 for r in trial_rows if r["error"])
    expected = config.permutation_count() * epochs
    # Multi-reference requests expand to one row per reference; count
    # distinct requests for reconciliation against expected samples.
    requests = len({r["request_id"] for r in trial_rows})
    by_request: dict[str, list[dict]] = {}
    for row in trial_rows:
        by_request.setdefault(row["request_id"], []).append(row)
    completed_requests = sum(
        1 for rows in by_request.values() if all(not row["error"] for row in rows)
    )
    error_requests = sum(
        1 for rows in by_request.values() if any(row["error"] for row in rows)
    )
    resolved_models = sorted(
        {r["resolved_model"] for r in trial_rows if r["resolved_model"]}
    )
    # Two requested aliases resolving to the same provider endpoint are
    # detectable here: the alias map groups requested ids by resolved id.
    alias_map: dict[str, list[str]] = {}
    for r in trial_rows:
        if r["resolved_model"]:
            requested = alias_map.setdefault(r["resolved_model"], [])
            if r["requested_model"] not in requested:
                requested.append(r["requested_model"])
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_identity": build_identity(),
        "execution_environment": execution_environment(),
        "protocol_role": config.protocol_role,
        "language_pairing_mode": config.language_pairing_mode,
        "config": config.to_dict(),
        "call_accounting": call_accounting(config, epochs),
        "generation_controls": GENERATION_CONTROLS,
        "requested_models": [m.inspect_model for m in config.models],
        "resolved_models": resolved_models,
        "model_alias_map": alias_map,
        # Provider-specific Inspect model args actually applied per model
        # (e.g. Together's stream=true); recorded so the request shape is
        # auditable alongside the shared generation controls.
        "model_args": {
            m.inspect_model: m.model_args
            for m in config.models
            if m.model_args
        },
        "counts": {
            "expected_samples": expected,
            "observed_requests": requests,
            "trial_rows": len(trial_rows),
            "completed": completed,
            "errors": errored,
            "missing": max(expected - requests, 0),
            "completed_requests": completed_requests,
            "error_requests": error_requests,
            "completed_reference_observations": completed,
            "error_reference_observations": errored,
        },
    }


def export_package(
    config: StudyConfig,
    passages: dict[str, dict[str, Passage]],
    log_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 1,
    run_id: str = "",
    started_at: str | None = None,
    ended_at: str | None = None,
) -> Path:
    """Write the full auditable result package to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = build_trial_rows(log_dir)
    manifest = build_run_manifest(
        config,
        trial_rows,
        epochs,
        run_id,
        started_at=started_at,
        ended_at=ended_at,
    )
    fixtures = build_source_fixtures(config, passages)

    def _write_json(name: str, data) -> None:
        (out_dir / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _write_jsonl(name: str, rows: list[dict]) -> None:
        (out_dir / name).write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, default=str) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    _write_json("run-manifest.json", manifest)
    _write_jsonl("trials.jsonl", trial_rows)
    _write_jsonl("source-fixtures.jsonl", fixtures)
    _write_json("method-configs.json", build_method_configs(config))
    _write_json("scoring-config.json", build_scoring_config())
    return out_dir
