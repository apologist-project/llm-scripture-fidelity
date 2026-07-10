"""Common interface for Bible text providers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from scripture_fidelity.references import Reference


class ProviderError(RuntimeError):
    """Raised when a Bible API request fails or returns unusable data."""


@dataclass(frozen=True)
class Verse:
    chapter: int
    number: int
    text: str


@dataclass(frozen=True)
class Passage:
    """The text of a Scripture reference in a specific translation."""

    reference: str  # display form, e.g. "John 3:16"
    translation_id: str  # study-level id, e.g. "BSB"
    verses: list[Verse] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Passage as a single string without verse numbers."""
        return _collapse(" ".join(v.text for v in self.verses))

    def numbered_text(self) -> str:
        """Passage with [chapter:verse] markers (used for tool output/RAG)."""
        parts = [f"[{v.chapter}:{v.number}] {v.text}" for v in self.verses]
        return _collapse(" ".join(parts))

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "translation_id": self.translation_id,
            "verses": [
                {"chapter": v.chapter, "number": v.number, "text": v.text}
                for v in self.verses
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "Passage":
        return Passage(
            reference=data["reference"],
            translation_id=data["translation_id"],
            verses=[
                Verse(chapter=v["chapter"], number=v["number"], text=v["text"])
                for v in data["verses"]
            ],
        )


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def verse_in_range(ref: Reference, chapter: int, verse: int) -> bool:
    """Whether (chapter, verse) falls within the reference."""
    if ref.is_chapter:
        return chapter == ref.chapter
    start = (ref.chapter, ref.verse)
    end_ch = ref.end_chapter or ref.chapter
    end_v = ref.end_verse if ref.end_verse is not None else ref.verse
    return start <= (chapter, verse) <= (end_ch, end_v)


class BibleProvider(ABC):
    """A client for one Bible text API."""

    name: str

    @abstractmethod
    async def get_passage(
        self, api_bible_id: str, ref: Reference, translation_id: str
    ) -> Passage:
        """Fetch the text of ``ref`` for the provider-specific Bible id."""

    @abstractmethod
    async def list_bibles(self, language: str | None = None) -> list[dict]:
        """List available Bibles as dicts with id/name/language keys."""
