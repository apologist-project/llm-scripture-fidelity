"""Study configuration loaded from .env JSON arrays (with CLI overrides)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

VALID_METHODS = (
    "unassisted",
    "rag",
    "tool_call",
    "buffer_transform",
    "buffer_transform_selection",
    "web_search",
)
VALID_APIS = ("ao_lab", "api_bible", "youversion")
VALID_PROVIDERS = ("openai", "anthropic", "google", "together", "xai", "mockllm")
VALID_PAIRING_MODES = ("matched", "crossed")
VALID_PROTOCOL_ROLES = ("diagnostic", "confirmatory", "robustness", "exploratory")
VALID_RIGHTS = ("open", "restricted", "unknown")

# Study provider name -> Inspect provider prefix (where they differ)
_INSPECT_PROVIDER_ALIASES = {"xai": "grok"}


class ConfigError(ValueError):
    """Raised when the study configuration is invalid."""


@dataclass(frozen=True)
class ReferenceConfig:
    ref: str
    type: str
    description: str = ""


@dataclass(frozen=True)
class TranslationConfig:
    id: str
    language: str
    api: str
    api_bible_id: str
    name: str = ""
    rights: str = "unknown"
    verification: str = ""

    @property
    def display_name(self) -> str:
        return self.name or self.id

    @property
    def source_key(self) -> str:
        """Stable composite source identity (provider + provider Bible id +
        study translation id), independent of display names."""
        return f"{self.api}:{self.api_bible_id}:{self.id}"


def fixture_id(translation: TranslationConfig, ref: str) -> str:
    """Stable composite fixture identity for one (source, reference) pair:
    source provider, provider Bible id, translation id, canonical reference."""
    from scripture_fidelity.references import parse_reference

    return f"{translation.source_key}:{parse_reference(ref).usfm()}"


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
    set_sizes: list[int] = field(default_factory=lambda: [1])
    language_pairing_mode: str = "matched"
    language_pairs: list[tuple[str, str]] = field(default_factory=list)
    protocol_role: str = "diagnostic"

    def variant_pairs(self) -> list[tuple[str, TranslationConfig]]:
        """The (prompt_language, translation) pairs this study will run.

        In matched mode only the declared pairs are generated; in crossed
        mode (an exploratory treatment) the full language x translation
        cross-product is generated.
        """
        if self.language_pairing_mode == "crossed":
            return [
                (lang, translation)
                for translation in self.translations
                for lang in self.languages
            ]
        by_id = {t.id: t for t in self.translations}
        return [
            (lang, by_id[tid])
            for lang, tid in self.language_pairs
            if lang in self.languages and tid in by_id
        ]

    def reference_sets(self, size: int) -> list[list[ReferenceConfig]]:
        """Chunk the references (in config order) into sets of ``size``.

        The final set may be smaller when the reference count is not an
        exact multiple. Size 1 reproduces single-reference samples.
        """
        return [
            self.references[i : i + size]
            for i in range(0, len(self.references), size)
        ]

    def sample_count(self) -> int:
        """Total samples per (method, translation, language, temp) variant,
        summed across all configured set sizes."""
        return sum(len(self.reference_sets(size)) for size in self.set_sizes)

    def variant_counts(self) -> dict[str, int]:
        return {
            "references": len(self.references),
            "set_sizes": len(self.set_sizes),
            "methods": len(self.methods),
            "translations": len(self.translations),
            "languages": len(self.languages),
            "language_pairs": len(self.variant_pairs()),
            "models": len(self.models),
            "temperatures": len(self.temperatures),
        }

    def permutation_count(self) -> int:
        return (
            self.sample_count()
            * len(self.methods)
            * len(self.variant_pairs())
            * len(self.models)
            * len(self.temperatures)
        )

    def to_dict(self) -> dict:
        return {
            "references": [vars(r) for r in self.references],
            "set_sizes": list(self.set_sizes),
            "methods": list(self.methods),
            "translations": [vars(t) for t in self.translations],
            "languages": list(self.languages),
            "language_pairing_mode": self.language_pairing_mode,
            "language_pairs": [list(p) for p in self.language_pairs],
            "protocol_role": self.protocol_role,
            "models": [vars(m) for m in self.models],
            "temperatures": list(self.temperatures),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudyConfig":
        return cls(
            references=[ReferenceConfig(**r) for r in data.get("references", [])],
            methods=list(data.get("methods", [])),
            translations=[
                TranslationConfig(**t) for t in data.get("translations", [])
            ],
            languages=list(data.get("languages", [])),
            models=[ModelConfig(**m) for m in data.get("models", [])],
            temperatures=[float(t) for t in data.get("temperatures", [])],
            set_sizes=list(data.get("set_sizes", [1])),
            language_pairing_mode=data.get("language_pairing_mode", "matched"),
            language_pairs=[tuple(p) for p in data.get("language_pairs", [])],
            protocol_role=data.get("protocol_role", "diagnostic"),
        )


def _load_json_env(name: str, default: list | None = None) -> list:
    raw = os.environ.get(name, "").strip()
    if not raw:
        if default is not None:
            return default
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
            ReferenceConfig(
                ref=item["ref"],
                type=item.get("type") or infer_type(parsed),
                description=item.get("description", ""),
            )
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
        rights = item.get("rights", "unknown")
        if rights not in VALID_RIGHTS:
            raise ConfigError(
                f"Unknown rights status {rights!r} (expected one of {VALID_RIGHTS})"
            )
        translations.append(
            TranslationConfig(
                id=item["id"],
                language=item["language"],
                api=item["api"],
                api_bible_id=str(item["api_bible_id"]),
                name=item.get("name", ""),
                rights=rights,
                verification=item.get("verification", ""),
            )
        )

    seen_ids: set[str] = set()
    for t in translations:
        if t.id in seen_ids:
            raise ConfigError(
                f"Duplicate translation id {t.id!r}: every TRANSLATIONS entry "
                "must have a unique id (e.g. rename to HIN_IRV / URD_IRV)"
            )
        seen_ids.add(t.id)

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

    set_sizes = []
    for size in _load_json_env("REFERENCE_SET_SIZES", default=[1]):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ConfigError(
                f"REFERENCE_SET_SIZES entries must be positive integers: {size!r}"
            )
        set_sizes.append(size)

    protocol_role = os.environ.get("PROTOCOL_ROLE", "").strip()
    if not protocol_role:
        raise ConfigError(
            "Missing required env var PROTOCOL_ROLE "
            f"(one of {VALID_PROTOCOL_ROLES})"
        )
    if protocol_role not in VALID_PROTOCOL_ROLES:
        raise ConfigError(
            f"Unknown PROTOCOL_ROLE {protocol_role!r} "
            f"(expected one of {VALID_PROTOCOL_ROLES})"
        )

    # The pairing mode must be declared explicitly: crossed prompts cannot
    # be selected accidentally through omission or a default.
    pairing_mode = os.environ.get("LANGUAGE_PAIRING_MODE", "").strip()
    if not pairing_mode:
        raise ConfigError(
            "Missing required env var LANGUAGE_PAIRING_MODE "
            f"(one of {VALID_PAIRING_MODES}); crossed prompts are never "
            "selected by default"
        )
    if pairing_mode not in VALID_PAIRING_MODES:
        raise ConfigError(
            f"Unknown LANGUAGE_PAIRING_MODE {pairing_mode!r} "
            f"(expected one of {VALID_PAIRING_MODES})"
        )

    language_pairs: list[tuple[str, str]] = []
    if pairing_mode == "matched":
        translation_ids = {t.id for t in translations}
        for pair in _load_json_env("LANGUAGE_PAIRS"):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ConfigError(
                    "LANGUAGE_PAIRS entries must be "
                    f'["<prompt_language>", "<translation_id>"] pairs: {pair!r}'
                )
            lang, tid = str(pair[0]), str(pair[1])
            if lang not in languages:
                raise ConfigError(
                    f"LANGUAGE_PAIRS language {lang!r} not in LANGUAGES"
                )
            if tid not in translation_ids:
                raise ConfigError(
                    f"LANGUAGE_PAIRS translation {tid!r} not in TRANSLATIONS"
                )
            language_pairs.append((lang, tid))
    elif protocol_role != "exploratory":
        raise ConfigError(
            "LANGUAGE_PAIRING_MODE 'crossed' is an exploratory treatment and "
            f"cannot be used with PROTOCOL_ROLE {protocol_role!r}; set "
            "PROTOCOL_ROLE=exploratory or use matched pairing"
        )

    if protocol_role == "confirmatory":
        wide = [
            name
            for name, values in [
                ("MODELS", models),
                ("TEMPERATURES", temperatures),
                ("REFERENCE_SET_SIZES", set_sizes),
            ]
            if len(values) > 1
        ]
        if wide:
            raise ConfigError(
                "PROTOCOL_ROLE 'confirmatory' does not allow grid search over "
                f"exploratory dimensions; restrict {wide} to a single value "
                "or use an exploratory/diagnostic role"
            )

    if "buffer_transform_selection" in methods:
        missing_desc = [r.ref for r in references if not r.description]
        if missing_desc:
            raise ConfigError(
                "The buffer_transform_selection method requires a 'description' "
                f"for every REFERENCES entry; missing for: {missing_desc}"
            )
        if set_sizes != [1]:
            raise ConfigError(
                "The buffer_transform_selection method only supports "
                "REFERENCE_SET_SIZES=[1]"
            )

    return StudyConfig(
        references=references,
        methods=methods,
        translations=translations,
        languages=languages,
        models=models,
        temperatures=temperatures,
        set_sizes=set_sizes,
        language_pairing_mode=pairing_mode,
        language_pairs=language_pairs,
        protocol_role=protocol_role,
    )
