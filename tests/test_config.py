"""Tests for .env-based study configuration loading."""

import pytest

from scripture_fidelity.config import ConfigError, load_config

VALID_ENV = {
    "REFERENCES": '[{"ref": "John 3:16", "type": "well_known_single"}, "Psalm 117"]',
    "METHODS": '["unassisted", "rag"]',
    "TRANSLATIONS": (
        '[{"id": "BSB", "name": "Berean Standard Bible", "language": "eng",'
        ' "api": "helloao", "api_bible_id": "BSB"}]'
    ),
    "LANGUAGES": '["eng"]',
    "MODELS": '[{"provider": "mockllm", "model": "model"}]',
    "TEMPERATURES": "[0.0, 0.7]",
}


def set_env(monkeypatch, **overrides):
    for key, value in {**VALID_ENV, **overrides}.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_valid_config(monkeypatch):
    set_env(monkeypatch)
    config = load_config()
    assert [r.ref for r in config.references] == ["John 3:16", "Psalm 117"]
    assert config.references[0].type == "well_known_single"
    assert config.references[1].type == "chapter"  # inferred
    assert config.methods == ["unassisted", "rag"]
    assert config.translations[0].display_name == "Berean Standard Bible"
    assert config.models[0].inspect_model == "mockllm/model"
    assert config.temperatures == [0.0, 0.7]
    assert config.permutation_count() == 2 * 2 * 1 * 1 * 1 * 2


def test_missing_var(monkeypatch):
    set_env(monkeypatch, METHODS=None)
    with pytest.raises(ConfigError, match="METHODS"):
        load_config()


def test_invalid_json(monkeypatch):
    set_env(monkeypatch, MODELS="not json")
    with pytest.raises(ConfigError, match="MODELS"):
        load_config()


def test_unknown_method(monkeypatch):
    set_env(monkeypatch, METHODS='["telepathy"]')
    with pytest.raises(ConfigError, match="telepathy"):
        load_config()


def test_all_methods_valid(monkeypatch):
    set_env(
        monkeypatch,
        METHODS='["unassisted", "rag", "tool_call", "output_buffer", "web_search"]',
    )
    config = load_config()
    assert "web_search" in config.methods


def test_unknown_api(monkeypatch):
    set_env(
        monkeypatch,
        TRANSLATIONS=(
            '[{"id": "X", "language": "eng", "api": "nope", "api_bible_id": "x"}]'
        ),
    )
    with pytest.raises(ConfigError, match="nope"):
        load_config()


def test_bad_reference(monkeypatch):
    set_env(monkeypatch, REFERENCES='["Hezekiah 3:16"]')
    with pytest.raises(Exception, match="Hezekiah"):
        load_config()


def test_translation_missing_field(monkeypatch):
    set_env(monkeypatch, TRANSLATIONS='[{"id": "BSB", "language": "eng"}]')
    with pytest.raises(ConfigError, match="api"):
        load_config()
