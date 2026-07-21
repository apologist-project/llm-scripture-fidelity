"""Client for the Crossway ESV API (api.esv.org).

Requires ESV_API_KEY. The API serves a single English translation (ESV), so
there is no provider-side Bible id — ``api_bible_id`` is ignored.
See https://api.esv.org/docs/
"""

from __future__ import annotations

import os
import re

import httpx

from scripture_fidelity.bible.base import BibleProvider, Passage, ProviderError, Verse
from scripture_fidelity.references import Reference

BASE_URL = "https://api.esv.org/v3"

# "[1] In the beginning ... [2] Now the earth ..."
_VERSE_MARKER_RE = re.compile(r"\[(\d+)\]")
_COPYRIGHT_RE = re.compile(r"\s*\(ESV\)\s*$")

_ESV_BIBLE = {
    "id": "",
    "name": "English Standard Version",
    "abbreviation": "ESV",
    "language": "eng",
}


def split_numbered_text(content: str, ref: Reference) -> list[Verse]:
    """Split ESV plain-text content on [n] verse markers."""
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
        text = content[m.end() : end].strip()
        if num <= prev_num and chapter < (ref.end_chapter or ref.chapter):
            chapter += 1
        prev_num = num
        if text:
            verses.append(Verse(chapter=chapter, number=num, text=text))
    return verses


class ESVProvider(BibleProvider):
    name = "esv"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ESV_API_KEY", "")

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError(
                "ESV_API_KEY is not set (required for esv provider)"
            )
        return {"Authorization": f"Token {self.api_key}"}

    async def _get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code != 200:
                raise ProviderError(
                    f"esv request failed ({resp.status_code}): {path}"
                )
            return resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"esv request error: {path}: {e}") from e

    async def get_passage(
        self, api_bible_id: str, ref: Reference, translation_id: str
    ) -> Passage:
        del api_bible_id  # single-translation API; no Bible id
        params = {
            "q": ref.display(),
            "include-passage-references": "false",
            "include-verse-numbers": "true",
            "include-first-verse-numbers": "true",
            "include-footnotes": "false",
            "include-footnote-body": "false",
            "include-headings": "false",
            "include-short-copyright": "false",
            "include-copyright": "false",
            "include-selahs": "true",
            "indent-paragraphs": "0",
            "indent-poetry": "false",
        }
        data = await self._get_json("/passage/text/", params)
        passages = data.get("passages") or []
        content = "\n".join(
            _COPYRIGHT_RE.sub("", p).strip() for p in passages if isinstance(p, str)
        )
        if not content:
            raise ProviderError(f"esv returned no content for {ref.display()}")
        verses = split_numbered_text(content, ref)
        if not verses:
            raise ProviderError(f"esv returned no verses for {ref.display()}")
        return Passage(
            reference=ref.display(), translation_id=translation_id, verses=verses
        )

    async def list_bibles(self, language: str | None = None) -> list[dict]:
        if language and language != "eng":
            return []
        return [dict(_ESV_BIBLE)]
