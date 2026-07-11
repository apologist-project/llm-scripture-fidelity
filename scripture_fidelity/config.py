"""Study configuration loaded from .env JSON arrays (with CLI overrides)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

VALID_METHODS = ("unassisted", "rag", "tool_call", "output_buffer", "web_search")
VALID_APIS = ("ao_lab", "api_bible", "youversion")
VALID_PROVIDERS = ("openai", "anthropic", "google", "together", "xai", "mockllm")

# Study provider name -> Inspect provider prefix (where they differ)
_INSPECT_PROVIDER_ALIASES = {"xai": "grok"}


class ConfigError(ValueError):
    """Raised when the study configuration is invalid."""


@dataclass(frozen=True)
class ReferenceConfig:
    ref: str
    type: str


@dataclass(frozen=True)
class TranslationConfig:
    id: str
    language: str
    api: str
    api_bible_id: str
    name: str = ""

    @property
    def display_name(self) -> str:
        return self.name or self.id


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str

    @property
    def inspect_model(self) -> str:
        provider = _INSPECT_PROVIDER_ALIASES.get(self.provider, self.provider)
        return f"{provider}/{self.model}"


@dataclass
class StudyConfig:
    references: list[ReferenceConfig] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    translations: list[TranslationConfig] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)

    def variant_counts(self) -> dict[str, int]:
        return {
            "references": len(self.references),
            "methods": len(self.methods),
            "translations": len(self.translations),
            "languages": len(self.languages),
            "models": len(self.models),
            "temperatures": len(self.temperatures),
        }

    def permutation_count(self) -> int:
        total = 1
        for n in self.variant_counts().values():
            total *= n
        return total

    def to_dict(self) -> dict:
        return {
            "references": [vars(r) for r in self.references],
            "methods": list(self.methods),
            "translations": [vars(t) for t in self.translations],
            "languages": list(self.languages),
            "models": [vars(m) for m in self.models],
            "temperatures": list(self.temperatures),
        }


def _load_json_env(name: str) -> list:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ConfigError(f"Missing required env var {name} (JSON array)")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Env var {name} is not valid JSON: {e}") from e
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Env var {name} must be a non-empty JSON array")
    return value


def load_config(env_file: str | Path | None = None) -> StudyConfig:
    """Load and validate the study configuration from .env / environment."""
    if env_file is not None:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv()

    from scripture_fidelity.references import infer_type, parse_reference

    references = []
    for item in _load_json_env("REFERENCES"):
        if isinstance(item, str):
            item = {"ref": item}
        if "ref" not in item:
            raise ConfigError(f"REFERENCES entry missing 'ref': {item}")
        parsed = parse_reference(item["ref"])  # fail fast on bad references
        references.append(
            ReferenceConfig(ref=item["ref"], type=item.get("type") or infer_type(parsed))
        )

    methods = _load_json_env("METHODS")
    for m in methods:
        if m not in VALID_METHODS:
            raise ConfigError(f"Unknown method {m!r} (expected one of {VALID_METHODS})")

    translations = []
    for item in _load_json_env("TRANSLATIONS"):
        missing = {"id", "language", "api", "api_bible_id"} - set(item)
        if missing:
            raise ConfigError(f"TRANSLATIONS entry missing {sorted(missing)}: {item}")
        if item["api"] not in VALID_APIS:
            raise ConfigError(
                f"Unknown Bible API {item['api']!r} (expected one of {VALID_APIS})"
            )
        translations.append(
            TranslationConfig(
                id=item["id"],
                language=item["language"],
                api=item["api"],
                api_bible_id=str(item["api_bible_id"]),
                name=item.get("name", ""),
            )
        )

    languages = [str(lang) for lang in _load_json_env("LANGUAGES")]

    models = []
    for item in _load_json_env("MODELS"):
        missing = {"provider", "model"} - set(item)
        if missing:
            raise ConfigError(f"MODELS entry missing {sorted(missing)}: {item}")
        if item["provider"] not in VALID_PROVIDERS:
            raise ConfigError(
                f"Unknown model provider {item['provider']!r} "
                f"(expected one of {VALID_PROVIDERS})"
            )
        models.append(ModelConfig(provider=item["provider"], model=item["model"]))

    temperatures = [float(t) for t in _load_json_env("TEMPERATURES")]

    return StudyConfig(
        references=references,
        methods=methods,
        translations=translations,
        languages=languages,
        models=models,
        temperatures=temperatures,
    )
