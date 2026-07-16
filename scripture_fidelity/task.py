"""Inspect task factory: one Task per (method, translation, language, temp)."""

from __future__ import annotations

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


def variant_name(
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float,
    set_size: int = 1,
) -> str:
    temp = f"{temperature:g}".replace(".", "_")
    name = f"{method}__{translation.id}__{language}__t{temp}"
    if set_size > 1:
        name += f"__set{set_size}"
    return name


def build_sample(
    ref: ReferenceConfig,
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float,
    passage: Passage,
    pairing_mode: str = "matched",
    protocol_role: str = "diagnostic",
) -> Sample:
    prompt = build_prompt(
        language=language,
        method=method,
        reference=ref.ref,
        translation_name=translation.display_name,
        translation_id=translation.id,
        context=passage.text if method == "rag" else "",
        description=ref.description,
    )
    return Sample(
        id=ref.ref,
        input=prompt,
        target=passage.text,
        metadata={
            "reference": ref.ref,
            "ref_type": ref.type,
            "reference_description": ref.description,
            "method": method,
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
        },
    )


def build_multi_sample(
    refs: list[ReferenceConfig],
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float,
    passages: dict[str, Passage],
    set_size: int,
    pairing_mode: str = "matched",
    protocol_role: str = "diagnostic",
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
    return Sample(
        id="; ".join(ref_strings),
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
        },
    )


def build_task(
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float,
    references: list[ReferenceConfig],
    passages: dict[str, Passage],
    service: PassageService,
    set_size: int = 1,
    pairing_mode: str = "matched",
    protocol_role: str = "diagnostic",
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
            )
            for i in range(0, len(references), set_size)
        ]
    name = variant_name(method, translation, language, temperature, set_size)
    return Task(
        dataset=MemoryDataset(samples=samples, name=name),
        solver=solver_chain(
            method, language, translation, service, multi=set_size > 1
        ),
        scorer=quotation_fidelity(),
        config=GenerateConfig(temperature=temperature),
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
