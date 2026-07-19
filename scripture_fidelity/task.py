"""Inspect task factory: one Task per (method, translation, language, temp)."""

from __future__ import annotations

import hashlib

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

from scripture_fidelity.bible.base import Passage
from scripture_fidelity.bible.service import PassageService
from scripture_fidelity.config import ReferenceConfig, TranslationConfig, fixture_id
from scripture_fidelity.prompts import build_multi_prompt, build_prompt
from scripture_fidelity.scoring import quotation_fidelity
from scripture_fidelity.solvers import solver_chain

SCORER_NAME = "quotation_fidelity"


def _neutral_caller_prompt(
    *, prompt_family: str, ref: ReferenceConfig, translation: TranslationConfig
) -> str:
    if prompt_family == "explicit_reference":
        return (
            f"Quote {ref.ref} from the {translation.display_name} exactly. "
            "Return only the requested source text."
        )
    if prompt_family == "contextual_description":
        return (
            f"Quote the passage from the {translation.display_name} described "
            f"as follows: {ref.description} Return only the requested source text."
        )
    raise ValueError(f"Unsupported neutral prompt family: {prompt_family}")


def _model_input(
    generated_prompt: str,
    *,
    method: str,
    prompt_override: str,
    source_document_override: str,
) -> tuple[str, str]:
    """Keep the caller request separate from source-supplied context."""
    caller_prompt = prompt_override or generated_prompt
    if method == "rag" and source_document_override:
        return caller_prompt, (
            "<authoritative_source>\n"
            f"{source_document_override}\n"
            "</authoritative_source>\n\n"
            "<user_request>\n"
            f"{caller_prompt}\n"
            "</user_request>"
        )
    return caller_prompt, caller_prompt


def variant_name(
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float | None,
    set_size: int = 1,
    prompt_family: str = "method_specific",
) -> str:
    temp = "default" if temperature is None else f"{temperature:g}".replace(".", "_")
    name = f"{method}__{translation.id}__{language}__t{temp}"
    if set_size > 1:
        name += f"__set{set_size}"
    if prompt_family != "method_specific":
        name += f"__{prompt_family}"
    return name


def build_sample(
    ref: ReferenceConfig,
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float | None,
    passage: Passage,
    pairing_mode: str = "matched",
    protocol_role: str = "diagnostic",
    prompt_override: str = "",
    request_context: dict | None = None,
    source_document_override: str = "",
    prompt_family: str = "method_specific",
) -> Sample:
    wrap_source_separately = method == "rag" and language == "eng"
    generated_prompt = build_prompt(
        language=language,
        method="unassisted" if wrap_source_separately else method,
        reference=ref.ref,
        translation_name=translation.display_name,
        translation_id=translation.id,
        context=(
            source_document_override or passage.text
            if method == "rag" and not wrap_source_separately
            else ""
        ),
        description=ref.description,
    )
    caller_prompt, prompt = _model_input(
        (
            generated_prompt
            if prompt_family == "method_specific"
            else _neutral_caller_prompt(
                prompt_family=prompt_family, ref=ref, translation=translation
            )
        ),
        method=method,
        prompt_override=prompt_override,
        source_document_override=(
            source_document_override or passage.text
            if wrap_source_separately
            else source_document_override
        ),
    )
    request_context = request_context or {}
    return Sample(
        id=request_context.get("scenario_id") or ref.ref,
        input=prompt,
        target=passage.text,
        metadata={
            "reference": ref.ref,
            "ref_type": ref.type,
            "reference_description": ref.description,
            "method": method,
            "prompt_family": prompt_family,
            "translation": translation.id,
            "translation_api": translation.api,
            "translation_bible_id": translation.api_bible_id,
            "fixture_id": fixture_id(translation, ref.ref),
            "text_language": translation.language,
            "prompt_language": language,
            "language_match": language == translation.language,
            "language_pairing_mode": pairing_mode,
            "protocol_role": protocol_role,
            "temperature": temperature,
            "set_size": 1,
            "ground_truth_verses": [v.text for v in passage.verses],
            "caller_request_id": request_context.get("request_id") or None,
            "scenario_id": request_context.get("scenario_id") or None,
            "protocol_version": request_context.get("protocol_version") or None,
            "repetition": int(request_context.get("repetition", 1)),
            "prompt_source": "caller" if prompt_override else "generated",
            "prompt_sha256": hashlib.sha256(caller_prompt.encode("utf-8")).hexdigest(),
            "effective_user_input_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "source_fixture_id_requested": (
                request_context.get("source_fixture_id") or None
            ),
            "source_document_supplied": bool(source_document_override),
            "source_document_sha256": (
                hashlib.sha256(source_document_override.encode("utf-8")).hexdigest()
                if source_document_override
                else None
            ),
        },
    )


def build_multi_sample(
    refs: list[ReferenceConfig],
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float | None,
    passages: dict[str, Passage],
    set_size: int,
    pairing_mode: str = "matched",
    protocol_role: str = "diagnostic",
    prompt_override: str = "",
    request_context: dict | None = None,
    source_document_override: str = "",
) -> Sample:
    """One sample asking for several references in a single prompt.

    ``set_size`` is the configured chunk size (the study arm), which may
    exceed len(refs) for the final remainder chunk."""
    ref_strings = [r.ref for r in refs]
    prompt = build_multi_prompt(
        language=language,
        method=method,
        references=ref_strings,
        translation_name=translation.display_name,
        translation_id=translation.id,
        contexts=(
            [(r, passages[r].text) for r in ref_strings] if method == "rag" else None
        ),
    )
    if prompt_override:
        prompt = prompt_override
    request_context = request_context or {}
    return Sample(
        id=request_context.get("scenario_id") or "; ".join(ref_strings),
        input=prompt,
        target="\n\n".join(passages[r].text for r in ref_strings),
        metadata={
            "reference": "; ".join(ref_strings),
            "ref_type": "set",
            "method": method,
            "translation": translation.id,
            "translation_api": translation.api,
            "translation_bible_id": translation.api_bible_id,
            "fixture_ids": [fixture_id(translation, r) for r in ref_strings],
            "text_language": translation.language,
            "prompt_language": language,
            "language_match": language == translation.language,
            "language_pairing_mode": pairing_mode,
            "protocol_role": protocol_role,
            "temperature": temperature,
            "set_size": set_size,
            "references": ref_strings,
            "ground_truth_texts": [passages[r].text for r in ref_strings],
            "ground_truth_verses_per_ref": [
                [v.text for v in passages[r].verses] for r in ref_strings
            ],
            "caller_request_id": request_context.get("request_id") or None,
            "scenario_id": request_context.get("scenario_id") or None,
            "protocol_version": request_context.get("protocol_version") or None,
            "repetition": int(request_context.get("repetition", 1)),
            "prompt_source": "caller" if prompt_override else "generated",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "effective_user_input_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "source_fixture_id_requested": (
                request_context.get("source_fixture_id") or None
            ),
        },
    )


def build_task(
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float | None,
    references: list[ReferenceConfig],
    passages: dict[str, Passage],
    service: PassageService,
    set_size: int = 1,
    pairing_mode: str = "matched",
    protocol_role: str = "diagnostic",
    prompt_override: str = "",
    request_context: dict | None = None,
    source_document_override: str = "",
    prompt_family: str = "method_specific",
) -> Task:
    """Build one Inspect task for a (method, translation, language, temp,
    set_size) variant. ``passages`` maps reference string -> Passage for
    this translation (prefetched ground truth). ``set_size`` > 1 chunks the
    references into multi-reference samples."""
    if set_size <= 1:
        samples = [
            build_sample(
                ref,
                method,
                translation,
                language,
                temperature,
                passages[ref.ref],
                pairing_mode=pairing_mode,
                protocol_role=protocol_role,
                prompt_override=prompt_override,
                request_context=request_context,
                source_document_override=source_document_override,
                prompt_family=prompt_family,
            )
            for ref in references
        ]
    else:
        samples = [
            build_multi_sample(
                references[i : i + set_size],
                method,
                translation,
                language,
                temperature,
                passages,
                set_size,
                pairing_mode=pairing_mode,
                protocol_role=protocol_role,
                prompt_override=prompt_override,
                request_context=request_context,
            )
            for i in range(0, len(references), set_size)
        ]
    name = variant_name(
        method, translation, language, temperature, set_size, prompt_family
    )
    return Task(
        dataset=MemoryDataset(samples=samples, name=name),
        solver=solver_chain(
            method, language, translation, service, multi=set_size > 1
        ),
        scorer=quotation_fidelity(),
        config=(
            GenerateConfig()
            if temperature is None
            else GenerateConfig(temperature=temperature)
        ),
        name=name,
        metadata={
            "method": method,
            "translation": translation.id,
            "translation_source_key": translation.source_key,
            "prompt_language": language,
            "language_match": language == translation.language,
            "language_pairing_mode": pairing_mode,
            "protocol_role": protocol_role,
            "temperature": temperature,
            "set_size": set_size,
        },
    )
