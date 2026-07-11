"""Client for the YouVersion Platform API (api.youversion.com).

Requires YOUVERSION_API_KEY (from platform.youversion.com). The passages
endpoint returns content whose exact shape can vary; this client extracts
text defensively and falls back to a single un-split verse block when
per-verse structure is unavailable.
"""

from __future__ import annotations

import os
import re

import httpx

from scripture_fidelity.bible.base import BibleProvider, Passage, ProviderError, Verse
from scripture_fidelity.references import Reference

BASE_URL = "https://api.youversion.com/v1"

_TAG_RE = re.compile(r"<[^>]+>")
_VERSE_LABEL_RE = re.compile(r"\{(\d+)\}|\[(\d+)\]")


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()


def extract_content(data: dict) -> str:
    """Pull the passage text out of a YouVersion passages response."""
    node = data.get("data", data)
    if isinstance(node, dict):
        for key in ("content", "text", "passage", "html"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return strip_html(value)
            if isinstance(value, dict):
                inner = extract_content({"data": value})
                if inner:
                    return inner
    return ""


class YouVersionProvider(BibleProvider):
    name = "youversion"

    def __init__(self, app_key: str | None = None):
        self.app_key = app_key or os.environ.get("YOUVERSION_API_KEY", "")

    def _headers(self) -> dict:
        if not self.app_key:
            raise ProviderError(
                "YOUVERSION_API_KEY is not set (required for youversion provider)"
            )
        return {"X-YVP-App-Key": self.app_key, "Accept": "application/json"}

    async def _get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code != 200:
                raise ProviderError(
                    f"youversion request failed ({resp.status_code}): {path}"
                )
            return resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"youversion request error: {path}: {e}") from e

    @staticmethod
    def _passage_id(ref: Reference) -> str:
        """YouVersion passage id: BOOK.CH, BOOK.CH.V, or BOOK.CH.V-V."""
        if ref.verse is None:
            return f"{ref.book}.{ref.chapter}"
        if ref.end_verse is None:
            return f"{ref.book}.{ref.chapter}.{ref.verse}"
        if ref.end_chapter is not None and ref.end_chapter != ref.chapter:
            raise ProviderError(
                f"youversion does not support cross-chapter ranges: {ref.display()}"
            )
        return f"{ref.book}.{ref.chapter}.{ref.verse}-{ref.end_verse}"

    async def get_passage(
        self, api_bible_id: str, ref: Reference, translation_id: str
    ) -> Passage:
        data = await self._get_json(
            f"/bibles/{api_bible_id}/passages/{self._passage_id(ref)}"
        )
        content = extract_content(data)
        if not content:
            raise ProviderError(
                f"youversion returned no content for {ref.display()} ({api_bible_id})"
            )
        verses = self._split_verses(content, ref)
        return Passage(
            reference=ref.display(), translation_id=translation_id, verses=verses
        )

    def _split_verses(self, content: str, ref: Reference) -> list[Verse]:
        matches = list(_VERSE_LABEL_RE.finditer(content))
        if not matches:
            start_verse = ref.verse if ref.verse is not None else 1
            return [
                Verse(chapter=ref.chapter, number=start_verse, text=content.strip())
            ]
        verses: list[Verse] = []
        chapter = ref.chapter
        prev_num = 0
        for i, m in enumerate(matches):
            num = int(m.group(1) or m.group(2))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[m.end():end].strip()
            if num <= prev_num and chapter < (ref.end_chapter or ref.chapter):
                chapter += 1
            prev_num = num
            if text:
                verses.append(Verse(chapter=chapter, number=num, text=text))
        return verses

    async def list_bibles(self, language: str | None = None) -> list[dict]:
        params = {"language_ranges": language} if language else None
        data = await self._get_json("/bibles", params)
        items = data.get("data", data.get("bibles", []))
        bibles = []
        for b in items if isinstance(items, list) else []:
            bibles.append(
                {
                    "id": b.get("id"),
                    "name": b.get("title") or b.get("name") or b.get("local_title"),
                    "abbreviation": b.get("abbreviation") or b.get("local_abbreviation"),
                    "language": (
                        b.get("language", {}).get("iso_639_3")
                        if isinstance(b.get("language"), dict)
                        else b.get("language")
                    ),
                }
            )
        return bibles
