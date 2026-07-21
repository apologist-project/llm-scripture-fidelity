"""Tests for the auditable result package export (AP-07/AP-08)."""

import json
from types import SimpleNamespace

import pytest

from scripture_fidelity.bible.base import Passage, Verse
from scripture_fidelity.config import (
    ModelConfig,
    ReferenceConfig,
    StudyConfig,
    TranslationConfig,
)
from scripture_fidelity.export import (
    ExportError,
    _expand_references,
    _structured_error,
    build_run_manifest,
    build_scoring_config,
    build_source_fixtures,
)

TRUTH = "For God so loved the world..."


def make_config(**overrides):
    defaults = dict(
        references=[ReferenceConfig(ref="John 3:16", type="single")],
        methods=["unassisted"],
        translations=[
            TranslationConfig(
                id="BSB", language="eng", api="ao_lab", api_bible_id="BSB"
            )
        ],
        languages=["eng"],
        models=[ModelConfig(provider="mockllm", model="model")],
        temperatures=[0.0],
        language_pairing_mode="matched",
        language_pairs=[("eng", "BSB")],
        protocol_role="diagnostic",
    )
    defaults.update(overrides)
    return StudyConfig(**defaults)


def make_passage(retrieved_at="2026-07-16T00:00:00+00:00"):
    return Passage(
        reference="John 3:16",
        translation_id="BSB",
        verses=[Verse(chapter=3, number=16, text=TRUTH)],
        retrieved_at=retrieved_at,
    )


def test_source_fixture_provenance_fields():
    config = make_config()
    translation = config.translations[0]
    fixtures = build_source_fixtures(
        config, {translation.source_key: {"John 3:16": make_passage()}}
    )
    assert len(fixtures) == 1
    row = fixtures[0]
    assert row["fixture_id"] == "ao_lab:BSB:BSB:JHN.3.16"
    assert row["source_api"] == "ao_lab"
    assert row["language"] == "eng"
    assert row["canonical_reference"] == "JHN.3.16"
    assert row["retrieved_at"] == "2026-07-16T00:00:00+00:00"
    assert row["text_sha256"] == make_passage().text_sha256
    assert row["normalized_text_sha256"] == make_passage().text_sha256
    assert row["normalization_version"]
    assert row["rights"] == "unknown"
    assert row["edition"] is None
    assert row["license_basis"] is None
    assert row["public_release"] is False
    assert row["text"] == TRUTH


def test_restricted_source_omits_text_but_keeps_hash():
    translation = TranslationConfig(
        id="NIV", language="eng", api="api_bible", api_bible_id="x",
        rights="restricted", verification="hash_only",
    )
    config = make_config(
        translations=[translation], language_pairs=[("eng", "NIV")]
    )
    fixtures = build_source_fixtures(
        config, {translation.source_key: {"John 3:16": make_passage()}}
    )
    row = fixtures[0]
    assert "text" not in row
    assert "verses" not in row
    assert row["text_sha256"] == make_passage().text_sha256
    assert row["verification"] == "hash_only"


def test_restricted_source_without_verification_blocks_export():
    translation = TranslationConfig(
        id="NIV", language="eng", api="api_bible", api_bible_id="x",
        rights="restricted",
    )
    config = make_config(
        translations=[translation], language_pairs=[("eng", "NIV")]
    )
    with pytest.raises(ExportError, match="verification"):
        build_source_fixtures(
            config, {translation.source_key: {"John 3:16": make_passage()}}
        )


def make_trial_row(**overrides):
    row = {
        "request_id": "run1:John 3:16:1",
        "requested_model": "mockllm/model",
        "resolved_model": "mockllm/model-2026-01-01",
        "reference": "John 3:16",
        "reference_index": 0,
        "error": None,
        "metrics": {"exact": 1.0},
    }
    row.update(overrides)
    return row


def test_manifest_counts_reconcile():
    config = make_config()
    rows = [
        make_trial_row(),
        make_trial_row(
            request_id="run1:Psalm 117:1",
            reference="Psalm 117",
            error="timeout",
        ),
    ]
    manifest = build_run_manifest(config, rows, epochs=2, run_id="run1")
    counts = manifest["counts"]
    assert counts["expected_samples"] == 2  # 1 sample/epoch x 2 epochs
    assert counts["observed_requests"] == 2
    assert counts["trial_rows"] == 2
    assert counts["completed"] == 1
    assert counts["errors"] == 1
    assert counts["missing"] == 0
    assert counts["completed_requests"] == 1
    assert counts["error_requests"] == 1
    assert manifest["build_identity"]["git_commit"]
    assert manifest["execution_environment"]["python"]
    assert manifest["protocol_role"] == "diagnostic"
    assert manifest["config"]["protocol_role"] == "diagnostic"


def test_manifest_detects_model_aliases():
    config = make_config()
    rows = [
        make_trial_row(requested_model="mockllm/latest"),
        make_trial_row(
            request_id="run1:Psalm 117:1", requested_model="mockllm/model"
        ),
    ]
    manifest = build_run_manifest(config, rows, epochs=1)
    aliases = manifest["model_alias_map"]["mockllm/model-2026-01-01"]
    assert sorted(aliases) == ["mockllm/latest", "mockllm/model"]


def test_manifest_records_provider_model_args():
    config = make_config(
        models=[
            ModelConfig(provider="together", model="Qwen/Qwen3.7-Max"),
            ModelConfig(provider="openai", model="gpt-5.4"),
        ]
    )
    manifest = build_run_manifest(config, [make_trial_row()], epochs=1)
    # Only models with provider-specific args appear; Together streaming is
    # recorded, OpenAI (no special args) is omitted.
    assert manifest["model_args"] == {
        "together/Qwen/Qwen3.7-Max": {"stream": True}
    }


def test_manifest_records_openrouter_provider_routing():
    routing = {
        "order": ["anthropic"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    config = make_config(
        models=[
            ModelConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-5",
                provider_routing=routing,
            )
        ]
    )
    manifest = build_run_manifest(config, [make_trial_row()], epochs=1)
    assert manifest["model_args"] == {
        "openrouter/anthropic/claude-sonnet-5": {"provider": routing}
    }


def test_multi_reference_rows_share_request_id_and_regroup(tmp_path):
    from scripture_fidelity.report.data import rows_from_export

    shared = {
        "request_id": "run1:John 3:16; Psalm 117:1",
        "requested_model": "mockllm/model",
        "resolved_model": "mockllm/model",
        "method": "unassisted",
        "translation": "BSB",
        "prompt_language": "eng",
        "language_match": True,
        "language_pairing_mode": "matched",
        "protocol_role": "diagnostic",
        "temperature": 0.0,
        "set_size": 2,
        "epoch": 1,
        "error": None,
        "metrics": {"exact": 0.5},
    }
    rows = [
        {**shared, "reference": "John 3:16", "reference_index": 0,
         "fixture_id": "ao_lab:BSB:BSB:JHN.3.16"},
        {**shared, "reference": "Psalm 117", "reference_index": 1,
         "fixture_id": "ao_lab:BSB:BSB:PSA.117"},
    ]
    (tmp_path / "trials.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    report_rows = rows_from_export(tmp_path)
    assert len(report_rows) == 1
    row = report_rows[0]
    assert row.reference == "John 3:16; Psalm 117"
    assert row.set_size == 2
    assert row.metrics == {"exact": 0.5}


def test_multi_reference_export_preserves_per_reference_metrics():
    md = {
        "references": ["John 3:16", "Psalm 117"],
        "fixture_ids": ["fixture-john", "fixture-psalm"],
        "set_size": 2,
    }
    request_metrics = {"exact": 0.5, "tool_used": 0.5}
    score_md = {
        "per_reference": [
            {
                "reference": "John 3:16",
                "answer": "correct",
                "metrics": {"exact": 1.0, "tool_used": 1.0},
            },
            {
                "reference": "Psalm 117",
                "answer": "wrong",
                "metrics": {"exact": 0.0, "tool_used": 0.0},
            },
        ]
    }
    common = {"request_id": "request-1", "request_metrics": request_metrics}

    rows = _expand_references(
        md,
        request_metrics,
        SimpleNamespace(answer="request answer"),
        score_md,
        common,
    )

    assert [row["trial_id"] for row in rows] == ["request-1:0", "request-1:1"]
    assert rows[0]["metrics"] == {"exact": 1.0, "tool_used": 1.0}
    assert rows[1]["metrics"] == {"exact": 0.0, "tool_used": 0.0}
    assert rows[0]["request_metrics"] == request_metrics
    assert rows[0]["answer"] == "correct"
    assert rows[1]["answer"] == "wrong"


def test_export_error_is_bounded_and_drops_traceback():
    error = _structured_error(
        "ModelGenerateError request body with private prompt traceback=full stack"
    )
    assert error["error_class"] == "ModelGenerateError"
    assert "traceback" not in error["message"]
    assert "private prompt" not in error["message"]


def test_export_error_redacts_likely_credentials():
    error = _structured_error(
        "ProviderError API_KEY=secret-value bearer:abcdefghijkl "
        "sk-live12345678"
    )
    assert "secret-value" not in error["message"]
    assert "abcdefghijkl" not in error["message"]
    assert "sk-live12345678" not in error["message"]


def test_report_recomputed_from_export_matches(tmp_path):
    from scripture_fidelity.report.data import aggregate, rows_from_export

    rows = [
        make_trial_row(
            method="unassisted", translation="BSB", prompt_language="eng",
            language_match=True, language_pairing_mode="matched",
            protocol_role="diagnostic", temperature=0.0, set_size=1,
            epoch=1, metrics={"exact": 1.0, "similarity": 1.0},
        ),
        make_trial_row(
            request_id="run1:Psalm 117:1", reference="Psalm 117",
            method="unassisted", translation="BSB", prompt_language="eng",
            language_match=True, language_pairing_mode="matched",
            protocol_role="diagnostic", temperature=0.0, set_size=1,
            epoch=1, metrics={"exact": 0.0, "similarity": 0.5},
        ),
    ]
    (tmp_path / "trials.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    report_rows = rows_from_export(tmp_path)
    assert len(report_rows) == 2
    [(key, means, count)] = aggregate(report_rows, ("method",))
    assert key == ("unassisted",)
    assert count == 2
    assert means["exact"] == 0.5
    assert means["similarity"] == 0.75


def test_scoring_config_declares_renderer_and_conjunction():
    config = build_scoring_config()
    assert "end_to_end_exact" in config
    assert "final_output_renderer" in config
    assert "quote_span_exact" in config["metrics"]
    assert "final_output_exact" in config["metrics"]


def _csv_trial_rows():
    from scripture_fidelity.report.data import TrialRow

    metrics = {
        "exact": 1.0, "normalized": 1.0, "similarity": 1.0, "cer": 0.0,
        "verse_coverage": 1.0, "final_output_exact": 1.0,
        "method_adherence": 1.0, "end_to_end_exact": 1.0,
    }
    return [
        TrialRow(
            model="openai/gpt-4o", method="unassisted", translation="BSB",
            language="eng", temperature=0.0, reference="John 3:16",
            ref_type="single", epoch=1, metrics=dict(metrics),
        ),
        TrialRow(
            model="openai/gpt-4o", method="rag", translation="BSB",
            language="eng", temperature=0.0, reference="Psalm 117",
            ref_type="chapter", epoch=1, metrics={**metrics, "exact": 0.0},
        ),
    ]


def test_write_csv_reports_matches_sections(tmp_path):
    import csv

    from scripture_fidelity.report.html_report import (
        build_sections,
        write_csv_reports,
    )

    rows = _csv_trial_rows()
    csv_dir = tmp_path / "csv"
    paths = write_csv_reports(rows, csv_dir)

    sections = build_sections(rows)
    # One CSV per table, all inside the csv/ folder.
    assert len(paths) == len(sections)
    assert all(p.parent == csv_dir and p.suffix == ".csv" for p in paths)

    # The detail-matrix CSV mirrors its section's headers and cell text exactly.
    detail = next(s for s in sections if s["title"].startswith("Detail matrix"))
    detail_csv = csv_dir / "detail-matrix-mean-over-iterations.csv"
    assert detail_csv in paths
    with detail_csv.open(encoding="utf-8", newline="") as f:
        read = list(csv.reader(f))
    assert read[0] == detail["headers"]
    assert read[1:] == [[c["text"] for c in r] for r in detail["rows"]]


def test_write_csv_reports_creates_folder(tmp_path):
    from scripture_fidelity.report.html_report import write_csv_reports

    csv_dir = tmp_path / "nested" / "csv"
    paths = write_csv_reports(_csv_trial_rows(), csv_dir)
    assert csv_dir.is_dir()
    assert paths


def test_write_csv_reports_one_file_per_table(tmp_path):
    from scripture_fidelity.report.html_report import (
        build_sections,
        write_csv_reports,
    )

    rows = _csv_trial_rows()
    paths = write_csv_reports(rows, tmp_path / "csv")
    # One CSV per HTML table, and all filenames distinct (no silent overwrite).
    assert len(paths) == len(build_sections(rows))
    assert len({p.name for p in paths}) == len(paths)
