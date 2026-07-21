"""Passage retrieval facade: provider registry + disk cache."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from scripture_fidelity.bible.api_bible import APIBibleProvider
from scripture_fidelity.bible.base import BibleProvider, Passage
from scripture_fidelity.bible.cache import DEFAULT_CACHE_DIR, PassageCache
from scripture_fidelity.bible.ao_lab import AOLabProvider
from scripture_fidelity.bible.esv import ESVProvider
from scripture_fidelity.bible.youversion import YouVersionProvider
from scripture_fidelity.config import TranslationConfig
from scripture_fidelity.references import Reference, parse_reference

_PROVIDERS: dict[str, type[BibleProvider]] = {
    "ao_lab": AOLabProvider,
    "api_bible": APIBibleProvider,
    "esv": ESVProvider,
    "youversion": YouVersionProvider,
}


def get_provider(name: str) -> BibleProvider:
    try:
        return _PROVIDERS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown Bible API provider: {name!r} (expected one of "
            f"{sorted(_PROVIDERS)})"
        ) from None


class PassageService:
    """Fetches passages for configured translations, with disk caching.

    The same fetched text serves as RAG context, tool-call output,
    buffer-transform replacement source, and the scoring ground truth.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self._cache = PassageCache(cache_dir)
        self._providers: dict[str, BibleProvider] = {}

    def _provider(self, name: str) -> BibleProvider:
        if name not in self._providers:
            self._providers[name] = get_provider(name)
        return self._providers[name]

    async def get(self, translation: TranslationConfig, ref: Reference) -> Passage:
        ref_key = ref.usfm()
        cached = self._cache.get(translation.api, translation.api_bible_id, ref_key)
        if cached is not None:
            return cached
        passage = await self._provider(translation.api).get_passage(
            translation.api_bible_id, ref, translation.id
        )
        if not passage.retrieved_at:
            passage = replace(
                passage,
                retrieved_at=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            )
        self._cache.put(translation.api, translation.api_bible_id, ref_key, passage)
        return passage

    async def get_by_ref_string(
        self, translation: TranslationConfig, ref_string: str
    ) -> Passage:
        return await self.get(translation, parse_reference(ref_string))
