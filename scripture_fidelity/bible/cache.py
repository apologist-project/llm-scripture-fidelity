"""Disk cache for fetched passages (also serves as the scoring ground truth)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripture_fidelity.bible.base import Passage

DEFAULT_CACHE_DIR = Path(".cache") / "passages"


class PassageCache:
    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)

    def _path(self, provider: str, api_bible_id: str, ref_key: str) -> Path:
        digest = hashlib.sha1(
            f"{provider}|{api_bible_id}|{ref_key}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, provider: str, api_bible_id: str, ref_key: str) -> Passage | None:
        path = self._path(provider, api_bible_id, ref_key)
        if not path.exists():
            return None
        try:
            return Passage.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            return None

    def put(
        self, provider: str, api_bible_id: str, ref_key: str, passage: Passage
    ) -> None:
        path = self._path(provider, api_bible_id, ref_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(passage.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
