"""Client for the AO Lab Free Use Bible API (bible.helloao.org). No key needed."""

from __future__ import annotations

import httpx

from scripture_fidelity.bible.base import (
    BibleProvider,
    Passage,
    ProviderError,
    Verse,
    verse_in_range,
)
from scripture_fidelity.references import Reference

BASE_URL = "https://bible.helloao.org/api"


def _flatten_verse_content(content: list) -> str:
    """Flatten AO Lab verse content (strings, formatted text, footnotes)."""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if "text" in item:  # FormattedText
                parts.append(item["text"])
            # noteId / heading / lineBreak entries carry no verse text
    return " ".join(p.strip() for p in parts if p and p.strip())


def parse_chapter_verses(chapter_json: dict) -> list[Verse]:
    """Extract Verse objects from an AO Lab chapter response."""
    chapter = chapter_json.get("chapter") or {}
    number = chapter.get("number")
    verses: list[Verse] = []
    for item in chapter.get("content", []):
        if isinstance(item, dict) and item.get("type") == "verse":
            verses.append(
                Verse(
                    chapter=number,
                    number=item["number"],
                    text=_flatten_verse_content(item.get("content", [])),
                )
            )
    return verses


class AOLabProvider(BibleProvider):
    name = "ao_lab"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    async def _get_json(self, url: str) -> dict | list:
        client = self._client or httpx.AsyncClient(timeout=30)
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise ProviderError(
                    f"ao_lab request failed ({resp.status_code}): {url}"
                )
            return resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"ao_lab request error: {url}: {e}") from e
        finally:
            if self._client is None:
                await client.aclose()

    async def get_passage(
        self, api_bible_id: str, ref: Reference, translation_id: str
    ) -> Passage:
        verses: list[Verse] = []
        for chapter_num in ref.chapters:
            url = f"{BASE_URL}/{api_bible_id}/{ref.book}/{chapter_num}.json"
            data = await self._get_json(url)
            chapter_verses = parse_chapter_verses(data)
            verses.extend(
                v
                for v in chapter_verses
                if verse_in_range(ref, chapter_num, v.number)
            )
        if not verses:
            raise ProviderError(
                f"ao_lab returned no verses for {ref.display()} ({api_bible_id})"
            )
        return Passage(
            reference=ref.display(), translation_id=translation_id, verses=verses
        )

    async def list_bibles(self, language: str | None = None) -> list[dict]:
        data = await self._get_json(f"{BASE_URL}/available_translations.json")
        bibles = [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "english_name": t.get("englishName"),
                "language": t.get("language"),
            }
            for t in data.get("translations", [])
        ]
        if language:
            bibles = [b for b in bibles if b["language"] == language]
        return sorted(bibles, key=lambda b: (b["language"] or "", b["id"] or ""))
