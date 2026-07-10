"""Tests for Scripture reference parsing."""

import pytest

from scripture_fidelity.references import (
    Reference,
    ReferenceError,
    infer_type,
    parse_reference,
)


def test_single_verse():
    ref = parse_reference("John 3:16")
    assert ref == Reference(book="JHN", chapter=3, verse=16)
    assert ref.display() == "John 3:16"
    assert ref.usfm() == "JHN.3.16"
    assert infer_type(ref) == "single"


def test_verse_range():
    ref = parse_reference("Romans 8:38-39")
    assert ref == Reference(book="ROM", chapter=8, verse=38, end_verse=39)
    assert ref.display() == "Romans 8:38-39"
    assert ref.usfm() == "ROM.8.38-ROM.8.39"
    assert infer_type(ref) == "range"


def test_whole_chapter():
    ref = parse_reference("Psalm 117")
    assert ref == Reference(book="PSA", chapter=117)
    assert ref.is_chapter
    assert ref.usfm() == "PSA.117"
    assert infer_type(ref) == "chapter"


def test_cross_chapter_range():
    ref = parse_reference("Luke 9:57-10:2")
    assert ref.book == "LUK"
    assert (ref.chapter, ref.verse) == (9, 57)
    assert (ref.end_chapter, ref.end_verse) == (10, 2)
    assert ref.chapters == [9, 10]
    assert ref.usfm() == "LUK.9.57-LUK.10.2"


@pytest.mark.parametrize(
    "text,book",
    [
        ("1 Jn. 1:9", "1JN"),
        ("First Samuel 3:10", "1SA"),
        ("ii kings 2:11", "2KI"),
        ("Song of Songs 2:1", "SNG"),
        ("Ps 23:1", "PSA"),
        ("JHN 3:16", "JHN"),
    ],
)
def test_book_aliases(text, book):
    assert parse_reference(text).book == book


def test_en_dash_range():
    ref = parse_reference("Romans 8:38\u201339")
    assert (ref.verse, ref.end_verse) == (38, 39)


def test_same_chapter_range_normalized():
    ref = parse_reference("John 3:16-3:17")
    assert ref.end_chapter is None
    assert ref.end_verse == 17


@pytest.mark.parametrize(
    "text",
    ["", "Hezekiah 3:16", "John", "John 3:17-16", "John 3:16-3:16"],
)
def test_invalid_references(text):
    with pytest.raises(ReferenceError):
        parse_reference(text)
