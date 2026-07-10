"""Inspect task factory: one Task per (method, translation, language, temp)."""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

from scripture_fidelity.bible.base import Passage
from scripture_fidelity.bible.service import PassageService
from scripture_fidelity.config import ReferenceConfig, TranslationConfig
from scripture_fidelity.prompts import build_prompt
from scripture_fidelity.scoring import quotation_fidelity
from scripture_fidelity.solvers import solver_chain

SCORER_NAME = "quotation_fidelity"


def variant_name(
    method: str, translation: TranslationConfig, language: str, temperature: float
) -> str:
    temp = f"{temperature:g}".replace(".", "_")
    return f"{method}__{translation.id}__{language}__t{temp}"


def build_sample(
    ref: ReferenceConfig,
    method: str,
    translation: TranslationConfig,
    language: str,
    temperature: float,
    passage: Passage,
) -> Sample:
    prompt = build_prompt(
        language=language,
        method=method,
        reference=ref.ref,
        translation_name=translation.display_name,
        translation_id=translation.id,
        context=passage.text if method == "rag" else "",
    )
    return Sample(
        id=ref.ref,
        input=prompt,
        target=passage.text,
        metadata={
            "reference": ref.ref,
            "ref_type": ref.type,
            "method": method,
            "translation": translation.id,
            "translation_api": translation.api,
            "text_language": translation.language,
            "prompt_language": language,
            "temperature": temperature,
            "ground_truth_verses": [v.text for v in passage.verses],
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
) -> Task:
    """Build one Inspect task for a (method, translation, language, temp)
    variant. ``passages`` maps reference string -> Passage for this
    translation (prefetched ground truth)."""
    samples = [
        build_sample(ref, method, translation, language, temperature, passages[ref.ref])
        for ref in references
    ]
    name = variant_name(method, translation, language, temperature)
    return Task(
        dataset=MemoryDataset(samples=samples, name=name),
        solver=solver_chain(method, language, translation, service),
        scorer=quotation_fidelity(),
        config=GenerateConfig(temperature=temperature),
        name=name,
        metadata={
            "method": method,
            "translation": translation.id,
            "prompt_language": language,
            "temperature": temperature,
        },
    )
