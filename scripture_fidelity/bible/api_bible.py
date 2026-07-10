"""Client for API.Bible (scripture.api.bible). Requires API_BIBLE_API_KEY."""

from __future__ import annotations

import os
import re

import httpx

from scripture_fidelity.bible.base import BibleProvider, Passage, ProviderError, Verse
from scripture_fidelity.references import Reference

BASE_URL = "https://api.scripture.api.bible/v1"

# "[1] In the beginning ... [2] Now the earth ..."
_VERSE_MARKER_RE = re.compile(r"\[(\d+)\]")


def split_numbered_text(content: str, ref: Reference) -> list[Verse]:
    """Split API.Bible text content on [n] verse markers.

    Chapter numbers are inferred: verse numbers reset when a new chapter
    starts (only relevant for cross-chapter ranges).
    """
    matches = list(_VERSE_MARKER_RE.finditer(content))
    if not matches:
        start_verse = ref.verse if ref.verse is not None else 1
        return [Verse(chapter=ref.chapter, number=start_verse, text=content.strip())]
    verses: list[Verse] = []
    chapter = ref.chapter
    prev_num = 0
    for i, m in enumerate(matches):
        num = int(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[m.end():end].strip()
        if num <= prev_num and chapter < (ref.end_chapter or ref.chapter):
            chapter += 1
        prev_num = num
        if text:
            verses.append(Verse(chapter=chapter, number=num, text=text))
    return verses


class APIBibleProvider(BibleProvider):
    name = "api_bible"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("API_BIBLE_API_KEY", "")

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError(
                "API_BIBLE_API_KEY is not set (required for api_bible provider)"
            )
        return {"api-key": self.api_key}

    async def _get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code != 200:
                raise ProviderError(
                    f"api_bible request failed ({resp.status_code}): {path}"
                )
            return resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"api_bible request error: {path}: {e}") from e

    async def get_passage(
        self, api_bible_id: str, ref: Reference, translation_id: str
    ) -> Passage:
        params = {
            "content-type": "text",
            "include-notes": "false",
            "include-titles": "false",
            "include-chapter-numbers": "false",
            "include-verse-numbers": "true",
            "include-verse-spans": "false",
        }
        data = await self._get_json(
            f"/bibles/{api_bible_id}/passages/{ref.usfm()}", params
        )
        content = (data.get("data") or {}).get("content", "")
        if not content or not content.strip():
            raise ProviderError(
                f"api_bible returned no content for {ref.display()} ({api_bible_id})"
            )
        verses = split_numbered_text(content, ref)
        return Passage(
            reference=ref.display(), translation_id=translation_id, verses=verses
        )

    async def list_bibles(self, language: str | None = None) -> list[dict]:
        params = {"language": language} if language else None
        data = await self._get_json("/bibles", params)
        return sorted(
            (
                {
                    "id": b.get("id"),
                    "name": b.get("name"),
                    "abbreviation": b.get("abbreviation"),
                    "language": (b.get("language") or {}).get("id"),
                }
                for b in data.get("data", [])
            ),
            key=lambda b: (b["language"] or "", b["abbreviation"] or ""),
        )
