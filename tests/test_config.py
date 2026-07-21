"""Tests for .env-based study configuration loading."""

import os

import pytest

from scripture_fidelity.config import ConfigError, load_config

VALID_ENV = {
    "REFERENCES": '[{"ref": "John 3:16", "type": "well_known_single"}, "Psalm 117"]',
    "METHODS": '["unassisted", "rag"]',
    "TRANSLATIONS": (
        '[{"id": "BSB", "name": "Berean Standard Bible", "language": "eng",'
        ' "api": "ao_lab", "api_bible_id": "BSB"}]'
    ),
    "LANGUAGES": '["eng"]',
    "LANGUAGE_PAIRING_MODE": "matched",
    "LANGUAGE_PAIRS": '[["eng", "BSB"]]',
    "PROTOCOL_ROLE": "diagnostic",
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
    monkeypatch.delenv("REFERENCE_SET_SIZES", raising=False)
    config = load_config()
    assert [r.ref for r in config.references] == ["John 3:16", "Psalm 117"]
    assert config.references[0].type == "well_known_single"
    assert config.references[1].type == "chapter"  # inferred
    assert config.methods == ["unassisted", "rag"]
    assert config.translations[0].display_name == "Berean Standard Bible"
    assert config.models[0].inspect_model == "mockllm/model"
    assert config.temperatures == [0.0, 0.7]
    assert config.prompt_families == ["method_specific"]
    assert config.set_sizes == [1]  # default
    assert config.language_pairing_mode == "matched"
    assert config.language_pairs == [("eng", "BSB")]
    assert config.protocol_role == "diagnostic"
    assert config.permutation_count() == 2 * 2 * 1 * 1 * 2


def test_prompt_families_are_crossed_as_preregistered_treatments(monkeypatch):
    set_env(
        monkeypatch,
        PROMPT_FAMILIES='["explicit_reference", "contextual_description"]',
    )
    config = load_config()

    assert config.prompt_families == [
        "explicit_reference",
        "contextual_description",
    ]
    assert config.permutation_count() == 2 * 2 * 1 * 1 * 2 * 2


def test_unknown_prompt_family_is_rejected(monkeypatch):
    set_env(monkeypatch, PROMPT_FAMILIES='["marketing_copy"]')

    with pytest.raises(ConfigError, match="PROMPT_FAMILIES"):
        load_config()


def test_together_model_enables_streaming(monkeypatch):
    set_env(
        monkeypatch,
        MODELS='[{"provider": "together", "model": "Qwen/Qwen3.7-Max"},'
        ' {"provider": "openai", "model": "gpt-5.4"}]',
    )
    config = load_config()
    together, openai = config.models
    assert together.inspect_model == "together/Qwen/Qwen3.7-Max"
    # Together requires streaming for some models; other providers unaffected.
    assert together.model_args == {"stream": True}
    assert openai.model_args == {}


def test_openrouter_model_applies_and_round_trips_locked_routing(monkeypatch):
    from scripture_fidelity.config import StudyConfig

    routing = (
        '{"order": ["anthropic"], "allow_fallbacks": false, '
        '"require_parameters": true, "data_collection": "deny"}'
    )
    set_env(
        monkeypatch,
        MODELS=(
            '[{"provider": "openrouter", '
            '"model": "anthropic/claude-sonnet-5", '
            f'"provider_routing": {routing}}}]'
        ),
    )
    config = load_config()
    assert config.models[0].model_args == {
        "provider": {
            "order": ["anthropic"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
    }
    assert StudyConfig.from_dict(config.to_dict()) == config


def test_provider_routing_is_rejected_for_direct_provider(monkeypatch):
    set_env(
        monkeypatch,
        MODELS=(
            '[{"provider": "openai", "model": "gpt-5.6-sol", '
            '"provider_routing": {"allow_fallbacks": false}}]'
        ),
    )
    with pytest.raises(ConfigError, match="only for openrouter"):
        load_config()


def test_xai_key_is_bridged_to_inspect_grok_provider(monkeypatch):
    set_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    load_config()

    assert os.environ["GROK_API_KEY"] == "test-xai-key"


def test_model_can_require_provider_default_temperature(monkeypatch):
    set_env(
        monkeypatch,
        MODELS=(
            '[{"provider": "openai", "model": "no-temperature", '
            '"supports_temperature": false}]'
        ),
        TEMPERATURES="[null]",
    )
    config = load_config()
    assert config.temperatures == [None]
    assert config.models[0].supports_temperature is False


def test_temperature_is_rejected_for_incompatible_model(monkeypatch):
    set_env(
        monkeypatch,
        MODELS=(
            '[{"provider": "openai", "model": "no-temperature", '
            '"supports_temperature": false}]'
        ),
        TEMPERATURES="[0.0]",
    )
    with pytest.raises(ConfigError, match="supports_temperature=false"):
        load_config()


def test_set_sizes_config(monkeypatch):
    set_env(monkeypatch, REFERENCE_SET_SIZES="[1, 2]")
    config = load_config()
    assert config.set_sizes == [1, 2]
    # size 1 -> two sets of one; size 2 -> one set of two
    assert config.reference_sets(1) == [
        [config.references[0]],
        [config.references[1]],
    ]
    assert config.reference_sets(2) == [config.references]
    assert config.sample_count() == 3
    assert config.permutation_count() == 3 * 2 * 1 * 1 * 2


def test_set_sizes_chunking_remainder(monkeypatch):
    set_env(
        monkeypatch,
        REFERENCES='["John 3:16", "Psalm 117", "Genesis 1:1"]',
        REFERENCE_SET_SIZES="[2]",
    )
    config = load_config()
    sets = config.reference_sets(2)
    assert [len(s) for s in sets] == [2, 1]


def test_invalid_set_sizes(monkeypatch):
    set_env(monkeypatch, REFERENCE_SET_SIZES="[0]")
    with pytest.raises(ConfigError, match="REFERENCE_SET_SIZES"):
        load_config()


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
        METHODS='["unassisted", "rag", "tool_call", "buffer_transform", "web_search"]',
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


def test_esv_translation_omits_api_bible_id(monkeypatch):
    set_env(
        monkeypatch,
        TRANSLATIONS=(
            '[{"id": "ESV", "name": "English Standard Version",'
            ' "language": "eng", "api": "esv"}]'
        ),
        LANGUAGE_PAIRS='[["eng", "ESV"]]',
    )
    config = load_config()
    assert config.translations[0].api == "esv"
    assert config.translations[0].api_bible_id == ""


def test_non_esv_translation_requires_api_bible_id(monkeypatch):
    set_env(
        monkeypatch,
        TRANSLATIONS=(
            '[{"id": "BSB", "name": "Berean Standard Bible",'
            ' "language": "eng", "api": "ao_lab"}]'
        ),
    )
    with pytest.raises(ConfigError, match="api_bible_id"):
        load_config()


def test_bad_reference(monkeypatch):
    set_env(monkeypatch, REFERENCES='["Hezekiah 3:16"]')
    with pytest.raises(Exception, match="Hezekiah"):
        load_config()


def test_translation_missing_field(monkeypatch):
    set_env(monkeypatch, TRANSLATIONS='[{"id": "BSB", "language": "eng"}]')
    with pytest.raises(ConfigError, match="api"):
        load_config()


TWO_TRANSLATIONS = (
    '[{"id": "HIN_IRV", "name": "Hindi IRV", "language": "hin",'
    ' "api": "ao_lab", "api_bible_id": "HINIRV"},'
    ' {"id": "URD_IRV", "name": "Urdu IRV", "language": "urd",'
    ' "api": "ao_lab", "api_bible_id": "urd_irv"}]'
)


def test_duplicate_translation_ids_rejected(monkeypatch):
    set_env(
        monkeypatch,
        TRANSLATIONS=(
            '[{"id": "IRV", "name": "Hindi IRV", "language": "hin",'
            ' "api": "ao_lab", "api_bible_id": "HINIRV"},'
            ' {"id": "IRV", "name": "Urdu IRV", "language": "urd",'
            ' "api": "ao_lab", "api_bible_id": "urd_irv"}]'
        ),
        LANGUAGES='["hin", "urd"]',
        LANGUAGE_PAIRS='[["hin", "IRV"]]',
    )
    with pytest.raises(ConfigError, match="[Dd]uplicate translation id"):
        load_config()


def test_fixture_identity_distinct_despite_similar_names(monkeypatch):
    from scripture_fidelity.config import fixture_id

    set_env(
        monkeypatch,
        TRANSLATIONS=TWO_TRANSLATIONS,
        LANGUAGES='["hin", "urd"]',
        LANGUAGE_PAIRS='[["hin", "HIN_IRV"], ["urd", "URD_IRV"]]',
    )
    config = load_config()
    hin, urd = config.translations
    assert fixture_id(hin, "John 3:16") != fixture_id(urd, "John 3:16")
    assert fixture_id(hin, "John 3:16") == "ao_lab:HINIRV:HIN_IRV:JHN.3.16"


def test_pairing_mode_required(monkeypatch):
    set_env(monkeypatch, LANGUAGE_PAIRING_MODE=None)
    with pytest.raises(ConfigError, match="LANGUAGE_PAIRING_MODE"):
        load_config()


def test_matched_mode_generates_only_declared_pairs(monkeypatch):
    set_env(
        monkeypatch,
        TRANSLATIONS=TWO_TRANSLATIONS,
        LANGUAGES='["eng", "hin", "urd"]',
        LANGUAGE_PAIRS='[["hin", "HIN_IRV"]]',
    )
    config = load_config()
    pairs = config.variant_pairs()
    assert [(lang, t.id) for lang, t in pairs] == [("hin", "HIN_IRV")]


def test_crossed_mode_requires_exploratory_role(monkeypatch):
    set_env(monkeypatch, LANGUAGE_PAIRING_MODE="crossed")
    with pytest.raises(ConfigError, match="exploratory"):
        load_config()


def test_crossed_mode_full_product(monkeypatch):
    set_env(
        monkeypatch,
        LANGUAGE_PAIRING_MODE="crossed",
        PROTOCOL_ROLE="exploratory",
        TRANSLATIONS=TWO_TRANSLATIONS,
        LANGUAGES='["eng", "zho"]',
    )
    config = load_config()
    assert len(config.variant_pairs()) == 4


def test_pair_referencing_unknown_translation(monkeypatch):
    set_env(monkeypatch, LANGUAGE_PAIRS='[["eng", "NOPE"]]')
    with pytest.raises(ConfigError, match="NOPE"):
        load_config()


def test_protocol_role_required(monkeypatch):
    set_env(monkeypatch, PROTOCOL_ROLE=None)
    with pytest.raises(ConfigError, match="PROTOCOL_ROLE"):
        load_config()


def test_protocol_role_invalid(monkeypatch):
    set_env(monkeypatch, PROTOCOL_ROLE="casual")
    with pytest.raises(ConfigError, match="casual"):
        load_config()


def test_full_grid_reports_484k_samples_per_epoch():
    """The proposed full exploratory grid is ~484,000 samples per epoch."""
    from scripture_fidelity.config import (
        ModelConfig,
        ReferenceConfig,
        StudyConfig,
        TranslationConfig,
    )

    refs = [ReferenceConfig(ref=f"John 3:{i}", type="single") for i in range(1, 9)]
    config = StudyConfig(
        references=refs,
        methods=["unassisted", "rag", "tool_call", "web_search", "buffer_transform"],
        translations=[
            TranslationConfig(
                id=f"T{i}", language="eng", api="ao_lab", api_bible_id=f"t{i}"
            )
            for i in range(16)
        ],
        languages=[f"l{i}" for i in range(11)],
        models=[ModelConfig(provider="mockllm", model=f"m{i}") for i in range(10)],
        temperatures=[0.0, 0.25, 0.5, 0.75, 1.0],
        set_sizes=[1, 3],
        language_pairing_mode="crossed",
        protocol_role="exploratory",
    )
    # 8 refs -> 8 singles + 3 sets of 3 = 11 samples per variant
    assert config.sample_count() == 11
    assert config.permutation_count() == 484_000


def test_selection_method_requires_descriptions(monkeypatch):
    set_env(monkeypatch, METHODS='["buffer_transform_selection"]')
    with pytest.raises(ConfigError, match="description"):
        load_config()


def test_selection_method_with_descriptions(monkeypatch):
    set_env(
        monkeypatch,
        REFERENCES=(
            '[{"ref": "John 3:16", "description": "the verse about God\'s love"},'
            ' {"ref": "Psalm 117", "description": "the shortest psalm"}]'
        ),
        METHODS='["buffer_transform_selection"]',
    )
    config = load_config()
    assert config.references[0].description == "the verse about God's love"


def test_selection_method_rejects_multi_reference_sets(monkeypatch):
    set_env(
        monkeypatch,
        REFERENCES=(
            '[{"ref": "John 3:16", "description": "d1"},'
            ' {"ref": "Psalm 117", "description": "d2"}]'
        ),
        METHODS='["buffer_transform_selection"]',
        REFERENCE_SET_SIZES="[1, 2]",
    )
    with pytest.raises(ConfigError, match="buffer_transform_selection"):
        load_config()


def test_translation_rights_and_verification(monkeypatch):
    set_env(
        monkeypatch,
        TRANSLATIONS=(
            '[{"id": "NIV", "language": "eng", "api": "api_bible",'
            ' "api_bible_id": "x", "rights": "restricted",'
            ' "verification": "hash_only", "edition": "2011",'
            ' "license_basis": "licensed API", "public_release": false}]'
        ),
        LANGUAGE_PAIRS='[["eng", "NIV"]]',
    )
    config = load_config()
    assert config.translations[0].rights == "restricted"
    assert config.translations[0].verification == "hash_only"
    assert config.translations[0].edition == "2011"
    assert config.translations[0].license_basis == "licensed API"
    assert config.translations[0].public_release is False


def test_translation_invalid_rights(monkeypatch):
    set_env(
        monkeypatch,
        TRANSLATIONS=(
            '[{"id": "NIV", "language": "eng", "api": "api_bible",'
            ' "api_bible_id": "x", "rights": "maybe"}]'
        ),
        LANGUAGE_PAIRS='[["eng", "NIV"]]',
    )
    with pytest.raises(ConfigError, match="maybe"):
        load_config()


def test_config_round_trips_through_dict(monkeypatch):
    from scripture_fidelity.config import StudyConfig

    set_env(monkeypatch)
    config = load_config()
    assert StudyConfig.from_dict(config.to_dict()) == config


def test_confirmatory_rejects_exploratory_dimensions(monkeypatch):
    set_env(monkeypatch, PROTOCOL_ROLE="confirmatory")  # TEMPERATURES has 2
    with pytest.raises(ConfigError, match="TEMPERATURES"):
        load_config()


def test_confirmatory_accepts_single_valued_grid(monkeypatch):
    set_env(
        monkeypatch,
        PROTOCOL_ROLE="confirmatory",
        TEMPERATURES="[0.0]",
        TRANSLATIONS=(
            '[{"id": "BSB", "name": "Berean Standard Bible", '
            '"language": "eng", "api": "ao_lab", "api_bible_id": "BSB", '
            '"rights": "open", "edition": "2025", '
            '"license_basis": "provider terms"}]'
        ),
    )
    config = load_config()
    assert config.protocol_role == "confirmatory"


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        ({}, "exactly one upstream"),
        (
            {
                "order": ["anthropic"],
                "allow_fallbacks": True,
                "require_parameters": True,
                "data_collection": "deny",
            },
            "allow_fallbacks=false",
        ),
        (
            {
                "order": ["anthropic"],
                "allow_fallbacks": False,
                "require_parameters": False,
                "data_collection": "deny",
            },
            "require_parameters=true",
        ),
        (
            {
                "order": ["anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "allow",
            },
            "data_collection='deny'",
        ),
    ],
)
def test_confirmatory_openrouter_requires_locked_routing(
    monkeypatch, routing, message
):
    import json

    model = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-5",
        "provider_routing": routing,
    }
    set_env(
        monkeypatch,
        PROTOCOL_ROLE="confirmatory",
        TEMPERATURES="[null]",
        MODELS=json.dumps([model]),
        TRANSLATIONS=(
            '[{"id": "BSB", "language": "eng", "api": "ao_lab", '
            '"api_bible_id": "BSB", "rights": "open", '
            '"edition": "2023", "license_basis": "public domain"}]'
        ),
    )
    with pytest.raises(ConfigError, match=message):
        load_config()


def test_confirmatory_requires_source_provenance(monkeypatch):
    set_env(monkeypatch, PROTOCOL_ROLE="confirmatory", TEMPERATURES="[0.0]")
    with pytest.raises(ConfigError, match="rights, edition, and license_basis"):
        load_config()


def test_call_accounting(monkeypatch):
    from scripture_fidelity.runner import call_accounting

    set_env(monkeypatch)
    config = load_config()
    accounting = call_accounting(config, epochs=3)
    # 2 refs x 2 methods x 1 pair x 1 model x 2 temps = 8 samples per epoch
    assert accounting["samples_per_epoch"] == 8
    assert accounting["planned_requests"] == 24
    # 2 methods x 1 pair x 1 model x 2 temps x 1 set size x 3 epochs
    assert accounting["observations_per_reference"] == 12
    assert (
        accounting["max_generation_attempts"]
        == 24 * (1 + accounting["retry_on_error"])
    )
