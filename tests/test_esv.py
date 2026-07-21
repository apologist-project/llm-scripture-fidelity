"""Unit tests for the ESV Bible provider helpers."""

import pytest

from scripture_fidelity.bible.esv import ESVProvider, split_numbered_text
from scripture_fidelity.references import parse_reference


def test_split_numbered_text_single_verse():
    ref = parse_reference("John 11:35")
    verses = split_numbered_text("[35] Jesus wept.", ref)
    assert len(verses) == 1
    assert verses[0].chapter == 11
    assert verses[0].number == 35
    assert verses[0].text == "Jesus wept."


def test_split_numbered_text_range():
    ref = parse_reference("John 3:16-17")
    verses = split_numbered_text(
        "[16] For God so loved the world... [17] For God did not send...",
        ref,
    )
    assert [(v.chapter, v.number) for v in verses] == [(3, 16), (3, 17)]


def test_split_numbered_text_cross_chapter():
    ref = parse_reference("Luke 9:62-10:2")
    verses = split_numbered_text(
        "[62] Jesus said... [1] After this... [2] And he said...",
        ref,
    )
    assert [(v.chapter, v.number) for v in verses] == [
        (9, 62),
        (10, 1),
        (10, 2),
    ]


@pytest.mark.asyncio
async def test_list_bibles_english_only():
    provider = ESVProvider(api_key="unused")
    assert len(await provider.list_bibles()) == 1
    assert len(await provider.list_bibles("eng")) == 1
    assert await provider.list_bibles("spa") == []
    bible = (await provider.list_bibles("eng"))[0]
    assert bible["abbreviation"] == "ESV"
    assert bible["language"] == "eng"
    assert bible["id"] == ""
