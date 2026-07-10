"""Parsing of Scripture references into a canonical, USFM-based form."""

from __future__ import annotations

import re
from dataclasses import dataclass

# (usfm, canonical name, aliases). Aliases are matched case-insensitively
# after collapsing whitespace and trailing periods ("1 Jn." -> "1 jn").
_BOOKS: list[tuple[str, str, list[str]]] = [
    ("GEN", "Genesis", ["gen", "ge", "gn"]),
    ("EXO", "Exodus", ["exod", "exo", "ex"]),
    ("LEV", "Leviticus", ["lev", "le", "lv"]),
    ("NUM", "Numbers", ["num", "nu", "nm", "nb"]),
    ("DEU", "Deuteronomy", ["deut", "deu", "dt"]),
    ("JOS", "Joshua", ["josh", "jos", "jsh"]),
    ("JDG", "Judges", ["judg", "jdg", "jg", "jdgs"]),
    ("RUT", "Ruth", ["rth", "rut", "ru"]),
    ("1SA", "1 Samuel", ["1 sam", "1sam", "1 sa", "1sa", "i samuel", "1st samuel", "first samuel"]),
    ("2SA", "2 Samuel", ["2 sam", "2sam", "2 sa", "2sa", "ii samuel", "2nd samuel", "second samuel"]),
    ("1KI", "1 Kings", ["1 kgs", "1kgs", "1 ki", "1ki", "i kings", "1st kings", "first kings"]),
    ("2KI", "2 Kings", ["2 kgs", "2kgs", "2 ki", "2ki", "ii kings", "2nd kings", "second kings"]),
    ("1CH", "1 Chronicles", ["1 chron", "1 chr", "1chr", "1 ch", "1ch", "i chronicles"]),
    ("2CH", "2 Chronicles", ["2 chron", "2 chr", "2chr", "2 ch", "2ch", "ii chronicles"]),
    ("EZR", "Ezra", ["ezr", "ez"]),
    ("NEH", "Nehemiah", ["neh", "ne"]),
    ("EST", "Esther", ["esth", "est", "es"]),
    ("JOB", "Job", ["jb"]),
    ("PSA", "Psalms", ["psalm", "pslm", "psa", "psm", "pss", "ps"]),
    ("PRO", "Proverbs", ["prov", "pro", "prv", "pr"]),
    ("ECC", "Ecclesiastes", ["eccles", "eccle", "ecc", "ec", "qoheleth"]),
    ("SNG", "Song of Solomon", ["song of songs", "song", "sos", "so", "sng", "canticles", "cant"]),
    ("ISA", "Isaiah", ["isa", "is"]),
    ("JER", "Jeremiah", ["jer", "je", "jr"]),
    ("LAM", "Lamentations", ["lam", "la"]),
    ("EZK", "Ezekiel", ["ezek", "eze", "ezk"]),
    ("DAN", "Daniel", ["dan", "da", "dn"]),
    ("HOS", "Hosea", ["hos", "ho"]),
    ("JOL", "Joel", ["joel", "jol", "jl"]),
    ("AMO", "Amos", ["amos", "amo", "am"]),
    ("OBA", "Obadiah", ["obad", "oba", "ob"]),
    ("JON", "Jonah", ["jnh", "jon"]),
    ("MIC", "Micah", ["mic", "mc"]),
    ("NAM", "Nahum", ["nah", "nam", "na"]),
    ("HAB", "Habakkuk", ["hab", "hb"]),
    ("ZEP", "Zephaniah", ["zeph", "zep", "zp"]),
    ("HAG", "Haggai", ["hag", "hg"]),
    ("ZEC", "Zechariah", ["zech", "zec", "zc"]),
    ("MAL", "Malachi", ["mal", "ml"]),
    ("MAT", "Matthew", ["matt", "mat", "mt"]),
    ("MRK", "Mark", ["mark", "mrk", "mar", "mk", "mr"]),
    ("LUK", "Luke", ["luke", "luk", "lk"]),
    ("JHN", "John", ["john", "jhn", "jn"]),
    ("ACT", "Acts", ["acts", "act", "ac"]),
    ("ROM", "Romans", ["rom", "ro", "rm"]),
    ("1CO", "1 Corinthians", ["1 cor", "1cor", "1 co", "1co", "i corinthians"]),
    ("2CO", "2 Corinthians", ["2 cor", "2cor", "2 co", "2co", "ii corinthians"]),
    ("GAL", "Galatians", ["gal", "ga"]),
    ("EPH", "Ephesians", ["eph", "ephes"]),
    ("PHP", "Philippians", ["phil", "php", "pp"]),
    ("COL", "Colossians", ["col", "co"]),
    ("1TH", "1 Thessalonians", ["1 thess", "1thess", "1 thes", "1 th", "1th", "i thessalonians"]),
    ("2TH", "2 Thessalonians", ["2 thess", "2thess", "2 thes", "2 th", "2th", "ii thessalonians"]),
    ("1TI", "1 Timothy", ["1 tim", "1tim", "1 ti", "1ti", "i timothy"]),
    ("2TI", "2 Timothy", ["2 tim", "2tim", "2 ti", "2ti", "ii timothy"]),
    ("TIT", "Titus", ["tit", "ti"]),
    ("PHM", "Philemon", ["philem", "phm", "pm"]),
    ("HEB", "Hebrews", ["heb"]),
    ("JAS", "James", ["jas", "jm"]),
    ("1PE", "1 Peter", ["1 pet", "1pet", "1 pe", "1pe", "1 pt", "i peter"]),
    ("2PE", "2 Peter", ["2 pet", "2pet", "2 pe", "2pe", "2 pt", "ii peter"]),
    ("1JN", "1 John", ["1 jn", "1jn", "1 jhn", "1 jo", "i john"]),
    ("2JN", "2 John", ["2 jn", "2jn", "2 jhn", "2 jo", "ii john"]),
    ("3JN", "3 John", ["3 jn", "3jn", "3 jhn", "3 jo", "iii john"]),
    ("JUD", "Jude", ["jude", "jud"]),
    ("REV", "Revelation", ["rev", "re", "apocalypse"]),
]

_ALIAS_TO_USFM: dict[str, str] = {}
for _usfm, _name, _aliases in _BOOKS:
    _ALIAS_TO_USFM[_usfm.lower()] = _usfm
    _ALIAS_TO_USFM[_name.lower()] = _usfm
    for _a in _aliases:
        _ALIAS_TO_USFM[_a] = _usfm

_USFM_TO_NAME = {usfm: name for usfm, name, _ in _BOOKS}


class ReferenceError(ValueError):
    """Raised when a Scripture reference cannot be parsed."""


@dataclass(frozen=True)
class Reference:
    """A canonical Scripture reference.

    ``verse is None`` means the whole chapter. ``end_chapter``/``end_verse``
    describe the (inclusive) end of a range; both are None for single verses
    and whole chapters.
    """

    book: str  # USFM code, e.g. "JHN"
    chapter: int
    verse: int | None = None
    end_chapter: int | None = None
    end_verse: int | None = None

    @property
    def book_name(self) -> str:
        return _USFM_TO_NAME[self.book]

    @property
    def is_chapter(self) -> bool:
        return self.verse is None

    @property
    def is_range(self) -> bool:
        return self.end_verse is not None

    @property
    def chapters(self) -> list[int]:
        """All chapter numbers this reference spans."""
        end = self.end_chapter or self.chapter
        return list(range(self.chapter, end + 1))

    def display(self) -> str:
        """Human-readable form, e.g. "John 3:16-17"."""
        s = f"{self.book_name} {self.chapter}"
        if self.verse is not None:
            s += f":{self.verse}"
            if self.end_verse is not None:
                if self.end_chapter is not None and self.end_chapter != self.chapter:
                    s += f"-{self.end_chapter}:{self.end_verse}"
                else:
                    s += f"-{self.end_verse}"
        return s

    def usfm(self) -> str:
        """USFM passage id, e.g. "JHN.3.16-JHN.3.19" or "PSA.117"."""
        if self.verse is None:
            return f"{self.book}.{self.chapter}"
        start = f"{self.book}.{self.chapter}.{self.verse}"
        if self.end_verse is None:
            return start
        end_ch = self.end_chapter or self.chapter
        return f"{start}-{self.book}.{end_ch}.{self.end_verse}"


def _lookup_book(name: str) -> str:
    key = re.sub(r"\s+", " ", name.strip().rstrip(".").lower())
    key = re.sub(r"^(1st|first)\b", "1", key)
    key = re.sub(r"^(2nd|second)\b", "2", key)
    key = re.sub(r"^(3rd|third)\b", "3", key)
    usfm = _ALIAS_TO_USFM.get(key)
    if usfm is None:
        raise ReferenceError(f"Unknown book name: {name!r}")
    return usfm


# "Book C", "Book C:V", "Book C:V-V", "Book C:V-C:V"
_REF_RE = re.compile(
    r"""^\s*
    (?P<book>(?:[1-3](?:st|nd|rd)?|[iI]{1,3}|[fF]irst|[sS]econd|[tT]hird)?\s*[A-Za-z][A-Za-z .]*?)
    \s+
    (?P<chapter>\d+)
    (?:\s*[:.]\s*(?P<verse>\d+)
        (?:\s*[-\u2013\u2014]\s*
            (?:(?P<end_chapter>\d+)\s*[:.]\s*)?
            (?P<end_verse>\d+)
        )?
    )?
    \s*$""",
    re.VERBOSE,
)


def parse_reference(text: str) -> Reference:
    """Parse a human-readable reference like "John 3:16-17" or "Psalm 117"."""
    m = _REF_RE.match(text)
    if m is None:
        raise ReferenceError(f"Cannot parse reference: {text!r}")
    book = _lookup_book(m.group("book"))
    chapter = int(m.group("chapter"))
    verse = int(m.group("verse")) if m.group("verse") else None
    end_chapter = int(m.group("end_chapter")) if m.group("end_chapter") else None
    end_verse = int(m.group("end_verse")) if m.group("end_verse") else None

    if end_verse is not None:
        eff_end_ch = end_chapter if end_chapter is not None else chapter
        if (eff_end_ch, end_verse) <= (chapter, verse):
            raise ReferenceError(f"Range end must come after start: {text!r}")
        if end_chapter is not None and end_chapter == chapter:
            end_chapter = None

    return Reference(
        book=book,
        chapter=chapter,
        verse=verse,
        end_chapter=end_chapter,
        end_verse=end_verse,
    )


def infer_type(ref: Reference) -> str:
    """Fallback grouping label when the config does not supply one."""
    if ref.is_chapter:
        return "chapter"
    if ref.is_range:
        return "range"
    return "single"
