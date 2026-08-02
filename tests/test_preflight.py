"""Tests for CLI dependency preflight checks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripture_fidelity.bible.base import Passage, Verse
from scripture_fidelity.config import (
    ModelConfig,
    ReferenceConfig,
    StudyConfig,
    TranslationConfig,
)
from scripture_fidelity.preflight import (
    CheckResult,
    highest_temperature,
    print_dependency_report,
    probe_temperature_for_model,
    run_dependency_checks,
)


def _config(**overrides) -> StudyConfig:
    base = dict(
        references=[ReferenceConfig(ref="John 3:16", type="single")],
        methods=["unassisted"],
        translations=[
            TranslationConfig(
                id="BSB",
                language="eng",
                api="ao_lab",
                api_bible_id="BSB",
                name="Berean Standard Bible",
            )
        ],
        languages=["eng"],
        models=[ModelConfig(provider="mockllm", model="model")],
        temperatures=[None],
        language_pairing_mode="matched",
        language_pairs=[("eng", "BSB")],
        protocol_role="diagnostic",
    )
    base.update(overrides)
    return StudyConfig(**base)


@pytest.fixture
def ok_passage() -> Passage:
    return Passage(
        reference="John 3:16",
        translation_id="BSB",
        verses=[Verse(chapter=3, number=16, text="For God so loved the world.")],
    )


def test_highest_temperature_picks_max_numeric():
    assert highest_temperature(_config(temperatures=[0.0, 0.5, 0.25])) == 0.5
    assert highest_temperature(_config(temperatures=[None])) is None


def test_probe_temperature_respects_supports_temperature_flag():
    config = _config(temperatures=[0.0, 0.7])
    capable = ModelConfig(provider="mockllm", model="model")
    omit = ModelConfig(
        provider="openai", model="no-temp", supports_temperature=False
    )
    assert probe_temperature_for_model(capable, config) == 0.7
    assert probe_temperature_for_model(omit, config) is None


def test_run_dependency_checks_collects_all_failures(monkeypatch):
    model_results = [
        CheckResult("model", "mockllm/model", False, "boom"),
        CheckResult("model", "openai/gpt-test", False, "auth"),
    ]
    translation_results = [
        CheckResult("translation", "BSB (ao_lab)", False, "missing"),
        CheckResult("translation", "ESV (esv)", True, "ok"),
    ]
    web_result = CheckResult("web_search", "Parallel.ai", False, "no key")

    async def fake_model(model, config):
        return model_results.pop(0)

    async def fake_translation(translation, ref_str):
        assert ref_str == "John 3:16"
        return translation_results.pop(0)

    async def fake_web(ref_str, translation):
        assert ref_str == "John 3:16"
        assert translation.id == "BSB"
        return web_result

    monkeypatch.setattr("scripture_fidelity.preflight._check_model", fake_model)
    monkeypatch.setattr(
        "scripture_fidelity.preflight._check_translation", fake_translation
    )
    monkeypatch.setattr("scripture_fidelity.preflight._check_web_search", fake_web)

    config = _config(
        methods=["unassisted", "web_search"],
        models=[
            ModelConfig(provider="mockllm", model="model"),
            ModelConfig(provider="openai", model="gpt-test"),
        ],
        translations=[
            TranslationConfig(
                id="BSB", language="eng", api="ao_lab", api_bible_id="BSB"
            ),
            TranslationConfig(id="ESV", language="eng", api="esv", api_bible_id=""),
        ],
    )
    results = run_dependency_checks(config)

    assert len(results) == 5
    assert sum(1 for r in results if not r.ok) == 4
    assert any(r.category == "web_search" for r in results)


def test_web_search_skipped_when_method_absent(monkeypatch):
    async def fake_model(model, config):
        return CheckResult("model", model.inspect_model, True, "ok")

    async def fake_translation(translation, ref_str):
        return CheckResult("translation", translation.id, True, "ok")

    called = {"web": False}

    async def fake_web(ref_str, translation):
        called["web"] = True
        return CheckResult("web_search", "Parallel.ai", True, "ok")

    monkeypatch.setattr("scripture_fidelity.preflight._check_model", fake_model)
    monkeypatch.setattr(
        "scripture_fidelity.preflight._check_translation", fake_translation
    )
    monkeypatch.setattr("scripture_fidelity.preflight._check_web_search", fake_web)

    results = run_dependency_checks(_config(methods=["rag", "tool_call"]))
    assert called["web"] is False
    assert all(r.category != "web_search" for r in results)


@pytest.mark.asyncio
async def test_check_model_uses_highest_temperature(monkeypatch):
    from scripture_fidelity import preflight
    import inspect_ai.model as model_mod

    output = SimpleNamespace(completion="ok")
    generate = AsyncMock(return_value=output)
    llm = SimpleNamespace(generate=generate)
    monkeypatch.setattr(model_mod, "get_model", MagicMock(return_value=llm))

    config = _config(temperatures=[0.0, 0.5])
    result = await preflight._check_model(
        ModelConfig(provider="mockllm", model="model"), config
    )
    assert result.ok
    assert "temperature=0.5" in result.detail
    assert "John 3:16" in result.detail
    kwargs = generate.await_args.kwargs
    assert kwargs["config"].temperature == 0.5


@pytest.mark.asyncio
async def test_check_model_omits_temperature_when_unsupported(monkeypatch):
    from scripture_fidelity import preflight
    import inspect_ai.model as model_mod

    output = SimpleNamespace(completion="ok")
    generate = AsyncMock(return_value=output)
    llm = SimpleNamespace(generate=generate)
    monkeypatch.setattr(model_mod, "get_model", MagicMock(return_value=llm))

    config = _config(temperatures=[0.0, 0.5])
    result = await preflight._check_model(
        ModelConfig(
            provider="openai", model="no-temp", supports_temperature=False
        ),
        config,
    )
    assert result.ok
    assert "provider default" in result.detail
    assert generate.await_args.kwargs["config"].temperature is None


@pytest.mark.asyncio
async def test_check_model_reports_empty_completion(monkeypatch):
    from scripture_fidelity import preflight
    import inspect_ai.model as model_mod

    output = SimpleNamespace(completion="   ")
    llm = SimpleNamespace(generate=AsyncMock(return_value=output))
    monkeypatch.setattr(model_mod, "get_model", MagicMock(return_value=llm))

    result = await preflight._check_model(
        ModelConfig(provider="mockllm", model="model"), _config()
    )
    assert not result.ok
    assert "empty" in result.detail


@pytest.mark.asyncio
async def test_check_model_captures_warnings(monkeypatch):
    from scripture_fidelity import preflight
    import inspect_ai.model as model_mod
    import warnings

    async def noisy_generate(*_args, **_kwargs):
        warnings.warn(
            "mockllm/model does not support the 'temperature' parameter",
            UserWarning,
        )
        return SimpleNamespace(completion="ok")

    llm = SimpleNamespace(generate=noisy_generate)
    monkeypatch.setattr(model_mod, "get_model", MagicMock(return_value=llm))

    result = await preflight._check_model(
        ModelConfig(provider="mockllm", model="model"),
        _config(temperatures=[0.7]),
    )
    assert result.ok
    assert result.has_warnings
    assert any("temperature" in w.casefold() for w in result.warnings)


@pytest.mark.asyncio
async def test_check_model_ignores_sibling_model_warnings(monkeypatch):
    from scripture_fidelity import preflight
    import inspect_ai.model as model_mod
    import logging

    async def noisy_generate(*_args, **_kwargs):
        logging.getLogger("inspect_ai.model._providers.anthropic").warning(
            "anthropic model 'claude-opus-5' does not support the "
            "'temperature' parameter (adaptive thinking only)."
        )
        return SimpleNamespace(completion="ok")

    llm = SimpleNamespace(generate=noisy_generate)
    monkeypatch.setattr(model_mod, "get_model", MagicMock(return_value=llm))

    result = await preflight._check_model(
        ModelConfig(provider="mockllm", model="model"),
        _config(temperatures=[0.7]),
    )
    assert result.ok
    assert not result.has_warnings


def test_format_error_prefers_api_message_over_request_dump():
    from scripture_fidelity.preflight import (
        _format_error,
        _looks_like_temperature_rejection,
    )

    cause = Exception(
        "Error code: 400 - {'error': {'message': "
        "\"Unsupported parameter: 'temperature' is not supported with this model.\""
        ", 'type': 'invalid_request_error'}}"
    )
    exc = RuntimeError(
        '\nRequest:\n{\n  "model": "gpt-5.6-terra", "temperature": 0.5\n}\n\n'
        "BadRequestError(\"...\")"
    )
    exc.__cause__ = cause
    detail = _format_error(exc)
    assert (
        "Unsupported parameter: 'temperature' is not supported with this model."
        in detail
    )
    assert "Request:" not in detail
    assert _looks_like_temperature_rejection(detail)


@pytest.mark.asyncio
async def test_check_translation_requires_verses(monkeypatch):
    from scripture_fidelity import preflight

    provider = SimpleNamespace(
        get_passage=AsyncMock(
            return_value=Passage(
                reference="John 3:16",
                translation_id="BSB",
                verses=[],
            )
        )
    )
    monkeypatch.setattr(preflight, "get_provider", lambda _name: provider)

    result = await preflight._check_translation(
        TranslationConfig(id="BSB", language="eng", api="ao_lab", api_bible_id="BSB"),
        "John 3:16",
    )
    assert not result.ok
    assert "no verses" in result.detail


@pytest.mark.asyncio
async def test_check_translation_success(monkeypatch, ok_passage):
    from scripture_fidelity import preflight

    provider = SimpleNamespace(get_passage=AsyncMock(return_value=ok_passage))
    monkeypatch.setattr(preflight, "get_provider", lambda _name: provider)

    result = await preflight._check_translation(
        TranslationConfig(id="BSB", language="eng", api="ao_lab", api_bible_id="BSB"),
        "John 3:16",
    )
    assert result.ok
    assert "John 3:16" in result.detail


@pytest.mark.asyncio
async def test_check_web_search_missing_key(monkeypatch):
    from scripture_fidelity import preflight

    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    result = await preflight._check_web_search(
        "John 3:16",
        TranslationConfig(id="BSB", language="eng", api="ao_lab", api_bible_id="BSB"),
    )
    assert not result.ok
    assert "PARALLEL_API_KEY" in result.detail


@pytest.mark.asyncio
async def test_check_web_search_success(monkeypatch):
    from scripture_fidelity import preflight

    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")

    client = SimpleNamespace(
        search=AsyncMock(
            return_value=SimpleNamespace(
                results=[SimpleNamespace(title="t", url="u", excerpts=["e"])]
            )
        ),
        close=AsyncMock(),
    )
    import parallel

    monkeypatch.setattr(parallel, "AsyncParallel", MagicMock(return_value=client))

    result = await preflight._check_web_search(
        "John 3:16",
        TranslationConfig(
            id="BSB",
            language="eng",
            api="ao_lab",
            api_bible_id="BSB",
            name="Berean Standard Bible",
        ),
    )
    assert result.ok
    assert "1 results" in result.detail
    assert "John 3:16" in result.detail
    client.search.assert_awaited()
    kwargs = client.search.await_args.kwargs
    assert "Berean Standard Bible" in kwargs["objective"]
    client.close.assert_awaited()


def test_print_dependency_report_counts_failures_not_warnings():
    from rich.console import Console

    console = Console(record=True, width=140)
    failures = print_dependency_report(
        [
            CheckResult("model", "a", True, "ok"),
            CheckResult(
                "model",
                "b",
                True,
                "ok",
                warnings=("temperature not supported",),
            ),
            CheckResult("translation", "c", False, "nope"),
        ],
        console,
    )
    assert failures == 1
    text = console.export_text()
    assert "WARN" in text
    assert "FAIL" in text
    assert "temperature not supported" in text


def test_cli_dry_run_returns_2_on_check_failure(monkeypatch):
    from scripture_fidelity import cli
    from scripture_fidelity.config import load_config

    set_env = {
        "REFERENCES": '["John 3:16"]',
        "METHODS": '["unassisted"]',
        "TRANSLATIONS": (
            '[{"id": "BSB", "language": "eng", "api": "ao_lab", '
            '"api_bible_id": "BSB"}]'
        ),
        "LANGUAGES": '["eng"]',
        "LANGUAGE_PAIRING_MODE": "matched",
        "LANGUAGE_PAIRS": '[["eng", "BSB"]]',
        "PROTOCOL_ROLE": "diagnostic",
        "MODELS": '[{"provider": "mockllm", "model": "model"}]',
        "TEMPERATURES": "[null]",
    }
    for key, value in set_env.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setattr(
        "scripture_fidelity.preflight.run_dependency_checks",
        lambda _config: [
            CheckResult("model", "mockllm/model", False, "probe failed"),
        ],
    )
    monkeypatch.setattr(
        "scripture_fidelity.config.load_config",
        lambda env_file=None: load_config(),
    )

    code = cli.main(["run", "--dry-run"])
    assert code == 2


def test_cli_real_run_aborts_before_study_on_check_failure(monkeypatch):
    from scripture_fidelity import cli
    from scripture_fidelity.config import load_config

    set_env = {
        "REFERENCES": '["John 3:16"]',
        "METHODS": '["unassisted"]',
        "TRANSLATIONS": (
            '[{"id": "BSB", "language": "eng", "api": "ao_lab", '
            '"api_bible_id": "BSB"}]'
        ),
        "LANGUAGES": '["eng"]',
        "LANGUAGE_PAIRING_MODE": "matched",
        "LANGUAGE_PAIRS": '[["eng", "BSB"]]',
        "PROTOCOL_ROLE": "diagnostic",
        "MODELS": '[{"provider": "mockllm", "model": "model"}]',
        "TEMPERATURES": "[null]",
    }
    for key, value in set_env.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setattr(
        "scripture_fidelity.preflight.run_dependency_checks",
        lambda _config: [
            CheckResult("translation", "BSB (ao_lab)", False, "unreachable"),
        ],
    )
    monkeypatch.setattr(
        "scripture_fidelity.config.load_config",
        lambda env_file=None: load_config(),
    )

    called = {"study": False}

    def fake_run_study(*_args, **_kwargs):
        called["study"] = True
        raise AssertionError("run_study should not be called")

    monkeypatch.setattr("scripture_fidelity.runner.run_study", fake_run_study)

    code = cli.main(["run"])
    assert code == 2
    assert called["study"] is False


def test_cli_warn_does_not_abort_real_run(monkeypatch):
    from scripture_fidelity import cli
    from scripture_fidelity.config import load_config

    set_env = {
        "REFERENCES": '["John 3:16"]',
        "METHODS": '["unassisted"]',
        "TRANSLATIONS": (
            '[{"id": "BSB", "language": "eng", "api": "ao_lab", '
            '"api_bible_id": "BSB"}]'
        ),
        "LANGUAGES": '["eng"]',
        "LANGUAGE_PAIRING_MODE": "matched",
        "LANGUAGE_PAIRS": '[["eng", "BSB"]]',
        "PROTOCOL_ROLE": "diagnostic",
        "MODELS": '[{"provider": "mockllm", "model": "model"}]',
        "TEMPERATURES": "[null]",
    }
    for key, value in set_env.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setattr(
        "scripture_fidelity.preflight.run_dependency_checks",
        lambda _config: [
            CheckResult(
                "model",
                "mockllm/model",
                True,
                "ok",
                warnings=("temperature not supported",),
            ),
        ],
    )
    monkeypatch.setattr(
        "scripture_fidelity.config.load_config",
        lambda env_file=None: load_config(),
    )

    called = {"study": False}

    def fake_run_study(*_args, **_kwargs):
        called["study"] = True
        from pathlib import Path

        return Path("/tmp/fake-logs")

    monkeypatch.setattr("scripture_fidelity.runner.run_study", fake_run_study)
    monkeypatch.setattr("scripture_fidelity.cli._emit_reports", lambda _log: 0)

    code = cli.main(["run"])
    assert code == 0
    assert called["study"] is True


def test_api_module_does_not_import_preflight():
    from pathlib import Path

    import scripture_fidelity.api as api

    source = Path(api.__file__).read_text(encoding="utf-8")
    assert "preflight" not in source
    assert "run_dependency_checks" not in source
